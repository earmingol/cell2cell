# -*- coding: utf-8 -*-

'''Ready-made objective functions for the ligand-receptor search.

`CorrelationObjective` is the one the search uses by default: it scores a candidate
set of pairs and correlates the resulting cell-cell interaction distances with a
reference matrix. See `base` for the contract a custom objective has to meet.
'''

from __future__ import absolute_import

import numpy as np
import pandas as pd
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


def correlation_fitness_batch(vectors, reference_vector, method='spearman', signed=False):
    '''
    `correlation_fitness` for a whole population at once.

    One `scipy.stats.spearmanr` call per candidate dominates the cost of a generation
    -- more than computing the interaction scores does. Spearman is Pearson on ranks,
    so ranking every candidate in one call and reducing with a single matrix-vector
    product gives the same numbers for the whole population at a fraction of the cost.

    Parameters
    ----------
    vectors : array-like
        Candidate vectors, of shape (n, length). One row per candidate.

    reference_vector : array-like
        The reference, of length `length`.

    method : str, default='spearman'
        'spearman' or 'pearson' take the vectorized path. A callable is called once
        per candidate, as `correlation_fitness` does, since nothing is known about it.

    signed : boolean, default=False
        Whether to keep the sign of the correlation.

    Returns
    -------
    fitness : numpy.ndarray
        One value per candidate. Higher is better.
    '''
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    reference_vector = np.asarray(reference_vector, dtype=float)

    if method == 'spearman':
        x = scipy.stats.rankdata(vectors, axis=1)
        y = scipy.stats.rankdata(reference_vector)
    elif method == 'pearson':
        x, y = vectors, reference_vector
    else:
        return np.array([correlation_fitness(vector, reference_vector, method=method,
                                             signed=signed) for vector in vectors])

    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean()
    # A candidate producing a constant vector has an undefined correlation, which
    # `correlation_fitness` turns into 0.0. The divide gives NaN here and
    # `nan_to_num` does the same, without scipy's warning.
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.divide(x @ y, np.sqrt((x ** 2).sum(axis=1) * (y ** 2).sum()))
    corr = np.nan_to_num(corr)
    return corr if signed else np.abs(corr)


def _as_square(matrix, name='reference_distances'):
    '''Validates a square reference, without making it symmetric.

    Used for `cci_type='directed'`, where the reference is expected to be asymmetric:
    averaging it with its transpose would destroy exactly the signal being looked for.
    '''
    values = np.asarray(matrix.values, dtype=float)
    if values.shape[0] != values.shape[1]:
        raise ValueError('`{}` must be a square matrix'.format(name))
    if list(matrix.index) != list(matrix.columns):
        raise ValueError('`{}` must have the same cells as rows and columns'.format(name))
    return matrix


def _as_block(matrix, row_cells, col_cells, name='reference_distances'):
    '''Validates a reference given as a block of rows against columns.

    The block is read by label, so the reference may be a larger matrix as long as it
    covers every cell asked for.
    '''
    missing_rows = [cell for cell in row_cells if cell not in matrix.index]
    missing_cols = [cell for cell in col_cells if cell not in matrix.columns]
    if missing_rows:
        raise ValueError('`{}` has no rows for {}'.format(name, missing_rows[:5]))
    if missing_cols:
        raise ValueError('`{}` has no columns for {}'.format(name, missing_cols[:5]))
    return matrix.loc[row_cells, col_cells]


def _comparison_mask(row_cells, col_cells, cci_type, include_self_interactions):
    '''Which entries of a (rows, columns) block are compared to the reference.

    Two things are decided here. Self-interactions are left out unless asked for,
    since the usual reference is a distance whose diagonal is a trivial zero. And
    under `cci_type='undirected'` an unordered pair of cells is a single quantity, so
    if it lands in the block twice -- which happens when `row_cells` and `col_cells`
    overlap -- it is compared once, rather than being weighted double against the
    pairs that land once.

    Parameters
    ----------
    row_cells, col_cells : list
        Cells on each axis of the block.

    cci_type : str
        'undirected' or 'directed'.

    include_self_interactions : boolean
        Whether to compare the entries where the same cell is on both axes.

    Returns
    -------
    mask : numpy.ndarray
        Boolean (rows, columns) array, True where the entry is compared.
    '''
    rows = np.asarray(list(row_cells), dtype=object)
    cols = np.asarray(list(col_cells), dtype=object)
    mask = np.ones((len(rows), len(cols)), dtype=bool)

    if cci_type == 'undirected':
        seen = set()
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                pair = frozenset((row, col))
                if pair in seen:
                    mask[i, j] = False
                else:
                    seen.add(pair)

    if not include_self_interactions:
        mask &= rows[:, None] != cols[None, :]
    return mask


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

    A 'score' column on the pool is taken as the weight of each interaction and
    multiplied into the weights a candidate selection produces. Weights of one, the
    usual case, make this a no-op.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression matrix, genes as rows and cells as columns.

    reference_distances : pandas.DataFrame
        The matrix the candidate's interaction distances are compared against.

        By default a square matrix over the same cells, of which only the upper
        triangle is read, so the diagonal is ignored. With `row_cells` and
        `col_cells` it is instead read as a block, cells on the rows against cells on
        the columns, and need only cover that block.

        Note that with `signed=False` an enrichment matrix works as well as a
        distance one -- the sign of the correlation flips but its absolute value does
        not. Under `cci_type='directed'` the matrix is read as senders against
        receivers and is not made symmetric.

    cutoff_setup : dict
        Cutoff setup, as in `cell2cell.analysis.initialize_interaction_space`.

    analysis_setup : dict
        With the keys 'communication_score', 'cci_score' and 'cci_type'.

        `cci_type='undirected'`, the default, makes the ligand-receptor pairs
        bidirectional and scores each pair of cells once, since A to B and B to A are
        then the same number. `'directed'` leaves the pairs unidirectional and scores
        the two orderings separately, so the reference is read as senders against
        receivers.

    included_cells : list, default=None
        Cells to use, on both axes. If None, those present in both `rnaseq_data` and
        `reference_distances`. Not compatible with `row_cells`/`col_cells`.

    row_cells, col_cells : list, default=None
        Cells to put on each axis, when the question is about one group of cells
        against another rather than about every pair. Give both or neither.

        A group of cell types against the rest, say: the reference is then that block
        and only that block is computed, which is the point -- the work drops from
        `cells^2` pairs to `len(row_cells) * len(col_cells)`.

        Under `cci_type='undirected'` the block is a sub-block of the symmetric
        matrix, so each entry still sums both orientations of every ligand-receptor
        pair and the two axes are interchangeable. Under `'directed'` the rows are
        the senders and the columns the receivers, and swapping them asks about
        signalling in the other direction.

        The two lists may overlap. An unordered cell pair landing in the block twice
        is still compared once when undirected, so it is not weighted double against
        the pairs that land once.

    include_self_interactions : boolean, default=False
        Whether to compare the entries where the same cell is on both axes.

        Left out by default, since the usual reference is a distance matrix whose
        diagonal is a trivial zero, and the interaction distances zero it too. Turn it
        on for a reference whose self-entries mean something -- a colocalization or
        enrichment score -- and the autocrine distances are kept rather than zeroed,
        so the entries carry real values on the same scale as the rest.

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

    One group of cell types against the rest, with a colocalization reference laid out
    as that block:

    >>> objective = c2c.analysis.CorrelationObjective(
    ...     rnaseq_data=rnaseq, reference_distances=colocalization,
    ...     row_cells=trophoblast_types, col_cells=other_types,
    ...     cutoff_setup={'type': 'constant_value', 'parameter': 0.05},
    ...     analysis_setup={'communication_score': 'expression_thresholding',
    ...                     'cci_score': 'jaccard', 'cci_type': 'undirected'})
    '''

    def __init__(self, rnaseq_data, reference_distances, cutoff_setup, analysis_setup,
                 included_cells=None, row_cells=None, col_cells=None,
                 include_self_interactions=False, correlation='spearman', signed=False,
                 fast=True, validate_fast=True, max_memory_mb=512,
                 interaction_columns=('A', 'B'), complex_sep=None,
                 complex_agg_method='min', verbose=False):
        cci_type = analysis_setup['cci_type']
        if cci_type not in ('undirected', 'directed'):
            raise ValueError("`cci_type` must be 'undirected' or 'directed', got {!r}"
                             .format(cci_type))

        self.rectangular = row_cells is not None or col_cells is not None
        if self.rectangular:
            if row_cells is None or col_cells is None:
                raise ValueError('Give both `row_cells` and `col_cells`, or neither.')
            if included_cells is not None:
                raise ValueError('`included_cells` puts the same cells on both axes. '
                                 'Use `row_cells` and `col_cells` instead, or drop them.')
            row_cells, col_cells = list(row_cells), list(col_cells)
            for label, subset in (('row_cells', row_cells), ('col_cells', col_cells)):
                if len(set(subset)) != len(subset):
                    raise ValueError('`{}` lists the same cell more than once, which '
                                     'would weight it twice.'.format(label))
            reference_block = _as_block(reference_distances, row_cells, col_cells)
        else:
            # The square form. Undirected is symmetrized as before; directed is
            # expected to be asymmetric, so it is only checked for shape.
            if cci_type == 'undirected':
                reference_distances = _as_symmetric(reference_distances)
            else:
                reference_distances = _as_square(reference_distances)
            if included_cells is None:
                included_cells = sorted(set(rnaseq_data.columns) & set(reference_distances.columns))
            included_cells = list(included_cells)
            if len(included_cells) < 3:
                raise ValueError('At least three cells are needed to correlate distances')
            row_cells = col_cells = included_cells
            reference_block = reference_distances.loc[included_cells, included_cells]

        self.row_cells = list(row_cells)
        self.col_cells = list(col_cells)
        self.include_self_interactions = include_self_interactions
        # The cells the interaction space has to be built over. For the square form
        # this is `included_cells` unchanged.
        self.space_cells = list(dict.fromkeys(self.row_cells + self.col_cells))
        self.mask = _comparison_mask(self.row_cells, self.col_cells, cci_type,
                                     include_self_interactions)

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

        self.reference_vector = np.asarray(reference_block.values, dtype=float)[self.mask]
        if self.reference_vector.size < 3:
            raise ValueError('At least three cell pairs are needed to correlate '
                             'distances, got {}.'.format(self.reference_vector.size))

        # Checked on the entries actually compared, so restricting the cells -- or
        # leaving the self-interactions out -- is a way out of it. Every correlation
        # against a reference holding NaN is NaN, which `correlation_fitness` turns
        # into 0.0: the search would then run to completion on a fitness that says
        # nothing about any candidate.
        if np.isnan(self.reference_vector).any():
            raise ValueError('`reference_distances` has missing values among the cell '
                             'pairs being compared. These are pairs whose distance was '
                             'never computed: restrict the cells with `included_cells` '
                             '(or `row_cells`/`col_cells`), or compute every pair '
                             'instead of passing `pairs` to the distance function.')

    def __call__(self, pool):
        return _BoundCorrelationObjective(self, pool)


class _BoundCorrelationObjective:
    '''A `CorrelationObjective` bound to one pool of candidate pairs.'''

    def __init__(self, parent, pool):
        self.parent = parent
        self.pool = pool
        verbose = parent.verbose
        setup = parent.analysis_setup

        self.cci_type = setup['cci_type']
        if self.cci_type == 'undirected':
            # The table and the provenance come from one call, so a candidate selection
            # can be expanded to per-row weights with no inference about how the
            # doubling went.
            ppi_for_space, self.source = bidirectional_ppi_with_index(
                ppi_data=pool, interaction_columns=parent.interaction_columns,
                verbose=verbose)
        else:
            # Directed keeps the ligand on the sender and the receptor on the receiver,
            # so the pairs are left as they are and each pool row is its own row.
            ppi_for_space = pool.copy()
            self.source = np.arange(len(pool))

        # A pool row with no row of its own in the bidirectional table could be
        # selected without changing anything, which would look like a pair that does
        # not matter rather than like the duplicate it is. Only reachable with
        # `deduplicate=False`, since the deduplicated pool has no repeated rows.
        missing = np.setdiff1d(np.arange(len(pool)), np.unique(self.source))
        if missing.size:
            raise ValueError(
                'Rows {} of the pool repeat an earlier row exactly, so they cannot be '
                'selected independently. Deduplicate the pairs before searching, or '
                'leave `deduplicate=True`.'.format(list(missing[:5])))

        # An interaction and its reverse produce the same two rows once the table is
        # doubled, so one row would have to belong to both candidates at once. Nothing
        # is doubled under 'directed', where the two directions are separate
        # quantities and listing both is the point.
        if self.cci_type == 'undirected':
            prot_a, prot_b = parent.interaction_columns
            pairs = list(zip(pool[prot_a], pool[prot_b]))
            pair_set = set(pairs)
            reciprocal = [position for position, (a, b) in enumerate(pairs)
                          if a != b and (b, a) in pair_set]
            if reciprocal:
                raise ValueError(
                    'The pool lists both directions of the same interaction (rows {}), '
                    'which cannot be selected independently once the table is made '
                    'bidirectional. Leave `deduplicate=True`, or collapse them with '
                    '`remove_ppi_bidirectionality`.'.format(reciprocal[:5]))
        self.interaction_space = InteractionSpace(
            rnaseq_data=parent.rnaseq_data[parent.space_cells],
            ppi_data=ppi_for_space,
            gene_cutoffs=parent.cutoff_setup,
            communication_score=setup['communication_score'],
            cci_score=setup['cci_score'],
            cci_type=setup['cci_type'],
            complex_sep=parent.complex_sep,
            complex_agg_method=parent.complex_agg_method,
            interaction_columns=parent.interaction_columns,
            verbose=verbose)


        # The interaction weight the pool itself carries, if any. A candidate selects
        # which interactions are active; the weight says how much each contributes, so
        # the two multiply rather than one standing in for the other. All ones -- the
        # usual case -- makes this a no-op.
        if 'score' in pool.columns:
            weights = np.asarray(pool['score'].values, dtype=float)
        else:
            weights = np.ones(len(pool), dtype=float)
        self.row_weights = weights[self.source]

        self.scorer = None
        if parent.fast:
            try:
                self.scorer = PreparedCCIScorer(self.interaction_space,
                                                cci_score=setup['cci_score'],
                                                max_memory_mb=parent.max_memory_mb,
                                                row_cells=parent.row_cells,
                                                col_cells=parent.col_cells)
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

    def _weights(self, masks):
        '''Per-row weights for a stack of candidate selections.'''
        return masks[:, self.source] * self.row_weights

    def reference_distance_vector(self, mask):
        '''Compared distances of one candidate, via the unmodified `InteractionSpace`.

        The distance is rebuilt here from the CCI matrix rather than read off
        `space.distance_matrix`, for two reasons. `distance_matrix` zeroes the
        self-interactions, which have to survive when they are being compared; and for
        the unbounded scores it regularizes by the mean of the whole matrix, whereas a
        block is regularized by the mean of the block. Over the full square matrix the
        two agree, so this is the same oracle as before for the square form.
        '''
        parent = self.parent
        weights = self._weights(np.atleast_2d(np.asarray(mask, dtype=float)))[0]
        space = self.interaction_space
        space.ppi_data['score'] = weights
        space.interaction_elements['ppi_score'] = space.ppi_data['score'].values
        space.compute_pairwise_cci_scores(use_ppi_score=True, verbose=False)

        block = space.interaction_elements['cci_matrix'].loc[parent.row_cells,
                                                             parent.col_cells]
        scores = np.asarray(block.values, dtype=float)
        if parent.analysis_setup['cci_score'] in ('count', 'icellnet'):
            mean = np.nanmean(scores)
            with np.errstate(divide='ignore', invalid='ignore'):
                distances = 1.0 - np.divide(scores, scores + mean)
        else:
            distances = 1.0 - scores
        # Entries left out of the comparison are never read, so the zeroing
        # `compute_pairwise_cci_scores` would apply to them makes no difference.
        return distances[parent.mask]

    def reference_value(self, mask):
        '''Fitness of one candidate through the unmodified `InteractionSpace` path.'''
        return self._fitness(self.reference_distance_vector(mask))

    def _distance_vectors(self, masks):
        '''Compared distances for a stack of candidates, through the scorer.'''
        weights = self._weights(np.atleast_2d(np.asarray(masks, dtype=float)))
        if self.parent.analysis_setup['cci_score'] == 'count':
            # 'count' asks whether an interaction is active, not how strongly, so the
            # weight only matters through its being non-zero. Passing it as it is would
            # trip the scorer's binary-weight check for a legitimately weighted pool.
            weights = (weights != 0).astype(float)
        distances = self.scorer.distance_batch(
            weights, zero_diagonal=not self.parent.include_self_interactions)
        return distances[:, self.parent.mask]

    def _validate(self):
        '''Checks the vectorized path against the reference one on a couple of probes.

        The comparison is on the distance matrices rather than on the fitness. The
        fitness is a rank correlation, so where two cell pairs are nearly tied a
        difference of a few ULP between the two paths flips their ranks and moves the
        correlation by far more than the distances themselves differ.
        '''
        n_pool = len(self.pool)
        probes = np.random.default_rng(0).integers(0, 2, size=(2, n_pool)).astype(float)
        fast_vectors = self._distance_vectors(probes)
        reference_vectors = np.array([self.reference_distance_vector(p) for p in probes])
        if not np.allclose(fast_vectors, reference_vectors, rtol=1e-9, atol=1e-9):
            worst = np.abs(fast_vectors - reference_vectors).max()
            raise RuntimeError(
                'The vectorized objective disagrees with the reference one (largest '
                'difference {:.3e}). Please report this, and use fast=False meanwhile.'
                .format(worst))

    def __call__(self, masks):
        masks = np.atleast_2d(np.asarray(masks, dtype=float))
        if self.scorer is None:
            return np.array([self.reference_value(mask) for mask in masks])

        vectors = self._distance_vectors(masks)
        return correlation_fitness_batch(vectors, self.parent.reference_vector,
                                         method=self.parent.correlation,
                                         signed=self.parent.signed)
