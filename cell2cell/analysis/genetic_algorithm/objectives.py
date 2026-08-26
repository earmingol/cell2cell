# -*- coding: utf-8 -*-

'''Ready-made objective functions for the ligand-receptor search.

`CorrelationObjective` is the one the search uses by default: it scores a candidate
set of pairs and correlates the resulting cell-cell interaction distances with a
reference matrix. See `base` for the contract a custom objective has to meet.
'''

from __future__ import absolute_import

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats

from cell2cell.core.interaction_space import InteractionSpace
from cell2cell.core.prepared_scorer import PreparedCCIScorer
from cell2cell.preprocessing.ppi import bidirectional_ppi_with_index


def correlation_fitness(distance_vector, reference_vector, method='spearman', signed=False):
    '''
    Correlation between a candidate's cell-cell distances and the reference.

    Parameters
    ----------
    distance_vector : array-like
        Condensed distance matrix from the candidate set of ligand-receptor pairs.

    reference_vector : array-like
        Condensed reference matrix, of the same length.

    method : str, default='spearman'
        'spearman' or 'pearson'.

    signed : boolean, default=False
        Whether to keep the sign of the correlation. The default takes the absolute
        value, which is usually what is wanted: it is not known in advance whether
        the search should favour pairs acting between cells that are *close* or
        pairs that mark cells which exclude each other, and both are real. Setting
        it to True restricts the search to positive associations only.

    Returns
    -------
    fitness : float
        Higher is better. NaN correlations, which arise when a candidate produces
        constant distances, become 0.0.
    '''
    if method == 'spearman':
        corr = scipy.stats.spearmanr(distance_vector, reference_vector)[0]
    elif method == 'pearson':
        corr = scipy.stats.pearsonr(distance_vector, reference_vector)[0]
    elif callable(method):
        corr = method(distance_vector, reference_vector)
    else:
        raise ValueError("`method` must be 'spearman', 'pearson' or a callable")
    corr = np.nan_to_num(corr)
    return float(corr) if signed else float(abs(corr))


def _as_symmetric(matrix):
    '''
    Validates a reference matrix and makes it exactly symmetric.

    `check_symmetry` compares with exact equality, which a matrix produced by a
    distance function often fails by a few ULP. Rather than rejecting those, they
    are averaged with their transpose; genuinely asymmetric input still raises.
    '''
    values = np.asarray(matrix.values, dtype=float)
    if values.shape[0] != values.shape[1]:
        raise ValueError('`reference_distances` must be a square matrix')
    if list(matrix.index) != list(matrix.columns):
        raise ValueError('`reference_distances` must have the same cells as rows and columns')
    if not np.allclose(values, values.T, rtol=1e-8, atol=1e-8, equal_nan=True):
        raise ValueError('`reference_distances` must be a symmetric matrix')
    return pd.DataFrame((values + values.T) / 2.0, index=matrix.index, columns=matrix.columns)


class CorrelationObjective:
    '''
    Correlates a candidate's cell-cell interaction distances with a reference matrix.

    This is the default objective, and the one the *C. elegans* analysis used: the
    fitness of a set of ligand-receptor pairs is how well the CCI distances it
    produces reproduce a reference distance between the same cells, such as
    physical distances from a 3D map or a spatial neighbourhood-enrichment matrix.

    An **objective factory** in the sense of `base`: calling it with a pool of
    candidate pairs returns an objective bound to that pool, having done all the
    per-pool work once.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression matrix, genes as rows and cells as columns.

    reference_distances : pandas.DataFrame
        Square, symmetric matrix over the same cells. Only the upper triangle is
        read, so the diagonal is ignored. Note that with `signed=False` an
        enrichment matrix works as well as a distance one -- the sign of the
        correlation flips but its absolute value does not.

    cutoff_setup : dict
        Cutoff setup, as in `cell2cell.analysis.initialize_interaction_space`.

    analysis_setup : dict
        With the keys 'communication_score', 'cci_score' and 'cci_type'. `cci_type`
        must be 'undirected'.

    included_cells : list, default=None
        Cells to use. If None, those present in both `rnaseq_data` and
        `reference_distances`.

    correlation : str or callable, default='spearman'
        Passed to `correlation_fitness`.

    signed : boolean, default=False
        Whether to keep the sign of the correlation. See `correlation_fitness`.

    fast : boolean, default=True
        Whether to evaluate with `PreparedCCIScorer`, which is equivalent but
        vectorized. False rebuilds the scores through `InteractionSpace` for every
        candidate, as the original implementation did.

    validate_fast : boolean, default=True
        Whether to check the vectorized path against the reference one on a couple
        of random candidates when the objective is built.

    max_memory_mb : float, default=512
        Budget for the scorer's precomputed outer products.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> objective = c2c.analysis.CorrelationObjective(
    ...     rnaseq_data=rnaseq, reference_distances=physical_distances,
    ...     cutoff_setup={'type': 'constant_value', 'parameter': 10},
    ...     analysis_setup={'communication_score': 'expression_thresholding',
    ...                     'cci_score': 'bray_curtis', 'cci_type': 'undirected'})
    >>> results = c2c.analysis.optimize_lr_pairs(ppi_data=lr_pairs, objective=objective)
    '''

    def __init__(self, rnaseq_data, reference_distances, cutoff_setup, analysis_setup,
                 included_cells=None, correlation='spearman', signed=False, fast=True,
                 validate_fast=True, max_memory_mb=512, interaction_columns=('A', 'B'),
                 complex_sep=None, complex_agg_method='min', verbose=False):
        if analysis_setup['cci_type'] != 'undirected':
            raise NotImplementedError(
                "Only 'undirected' interactions are supported, because the objective "
                "compares condensed symmetric matrices.")

        reference_distances = _as_symmetric(reference_distances)
        if included_cells is None:
            included_cells = sorted(set(rnaseq_data.columns) & set(reference_distances.columns))
        included_cells = list(included_cells)
        if len(included_cells) < 3:
            raise ValueError('At least three cells are needed to correlate distances')

        self.rnaseq_data = rnaseq_data
        self.reference_distances = reference_distances
        self.cutoff_setup = cutoff_setup
        self.analysis_setup = analysis_setup
        self.included_cells = included_cells
        self.correlation = correlation
        self.signed = signed
        self.fast = fast
        self.validate_fast = validate_fast
        self.max_memory_mb = max_memory_mb
        self.interaction_columns = interaction_columns
        self.complex_sep = complex_sep
        self.complex_agg_method = complex_agg_method
        self.verbose = verbose

        self.reference_vector = scipy.spatial.distance.squareform(
            np.asarray(reference_distances.loc[included_cells, included_cells].values,
                       dtype=float), checks=False)

    def __call__(self, pool):
        return _BoundCorrelationObjective(self, pool)


class _BoundCorrelationObjective:
    '''A `CorrelationObjective` bound to one pool of candidate pairs.'''

    def __init__(self, parent, pool):
        self.parent = parent
        self.pool = pool
        verbose = parent.verbose
        setup = parent.analysis_setup

        # The table and the provenance come from one call, so a candidate selection can
        # be expanded to per-row weights with no inference about how the doubling went.
        bi_ppi_data, self.source = bidirectional_ppi_with_index(
            ppi_data=pool, interaction_columns=parent.interaction_columns, verbose=verbose)
        self.interaction_space = InteractionSpace(
            rnaseq_data=parent.rnaseq_data[parent.included_cells],
            ppi_data=bi_ppi_data,
            gene_cutoffs=parent.cutoff_setup,
            communication_score=setup['communication_score'],
            cci_score=setup['cci_score'],
            cci_type=setup['cci_type'],
            complex_sep=parent.complex_sep,
            complex_agg_method=parent.complex_agg_method,
            interaction_columns=parent.interaction_columns,
            verbose=verbose)

        space_cells = list(self.interaction_space.interaction_elements['cell_names'])
        self.take = [space_cells.index(c) for c in parent.included_cells]

        self.scorer = None
        if parent.fast:
            try:
                self.scorer = PreparedCCIScorer(self.interaction_space,
                                                cci_score=setup['cci_score'],
                                                max_memory_mb=parent.max_memory_mb)
            except NotImplementedError:
                if verbose:
                    print('Falling back to the reference objective for this CCI score')
            # `count` treats a NaN product as active, which the scorer's zero
            # substitution does not reproduce. Only this score is affected.
            if self.scorer is not None and self.scorer._has_nans and setup['cci_score'] == 'count':
                self.scorer = None

        if self.scorer is not None and parent.validate_fast:
            self._validate()

    def _fitness(self, vector):
        return correlation_fitness(vector, self.parent.reference_vector,
                                   method=self.parent.correlation, signed=self.parent.signed)

    def reference_value(self, mask):
        '''Fitness of one candidate through the unmodified `InteractionSpace` path.'''
        weights = np.asarray(mask, dtype=float)[self.source]
        space = self.interaction_space
        space.ppi_data['score'] = weights
        space.interaction_elements['ppi_score'] = space.ppi_data['score'].values
        space.compute_pairwise_cci_scores(use_ppi_score=True, verbose=False)
        distances = space.distance_matrix.loc[self.parent.included_cells,
                                              self.parent.included_cells]
        vector = scipy.spatial.distance.squareform(np.asarray(distances.values, dtype=float),
                                                   checks=False)
        return self._fitness(vector)

    def _validate(self):
        n_pool = len(self.pool)
        probes = np.random.default_rng(0).integers(0, 2, size=(2, n_pool)).astype(float)
        fast_values = self(probes)
        reference_values = np.array([self.reference_value(p) for p in probes])
        if not np.allclose(fast_values, reference_values, rtol=1e-9, atol=1e-9):
            raise RuntimeError(
                'The vectorized objective disagrees with the reference one ({} vs {}). '
                'Please report this, and use fast=False meanwhile.'
                .format(fast_values, reference_values))

    def __call__(self, masks):
        masks = np.atleast_2d(np.asarray(masks, dtype=float))
        if self.scorer is None:
            return np.array([self.reference_value(mask) for mask in masks])

        weights = masks[:, self.source]
        distances = self.scorer.distance_batch(weights)[:, self.take][:, :, self.take]
        out = np.empty(len(masks))
        for n, matrix in enumerate(distances):
            out[n] = self._fitness(scipy.spatial.distance.squareform(matrix, checks=False))
        return out
