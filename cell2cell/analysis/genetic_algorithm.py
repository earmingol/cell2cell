# -*- coding: utf-8 -*-

'''Selection of ligand-receptor pairs with a genetic algorithm.

This module reimplements, as part of the package, the analysis in
https://github.com/LewisLabUCSD/Celegans-cell2cell (`code/genetic_algorithm.py`),
which searches for the subset of ligand-receptor pairs whose cell-cell interaction
scores best reproduce a reference distance between cells, for example the physical
distances measured in a 3D map.

Reference
---------
Armingol E, Ghaddar A, Joshi CJ, Baghdassarian H, Shamie I, Chan J, et al. (2022)
Inferring a spatial code of cell-cell interactions across a whole animal body.
PLOS Computational Biology 18(11): e1010715.
https://doi.org/10.1371/journal.pcbi.1010715

The objective function is the one used there: the absolute Spearman correlation
between the CCI distance matrix and the reference distance matrix. The original
implementation used `pyevolve`, which only supports Python 2; this one uses
`pygad`, and evaluates the objective in a vectorized way that is orders of
magnitude faster (see `optimize_lr_pairs` for the details).
'''

from __future__ import absolute_import

import warnings

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats

from types import ModuleType

from cell2cell.clustering.cluster_interactions import get_clusters_from_linkage
from cell2cell.core.interaction_space import InteractionSpace
from cell2cell.preprocessing.manipulate_dataframes import check_symmetry
from cell2cell.preprocessing.ppi import bidirectional_ppi_for_cci, remove_ppi_bidirectionality


# CCI scores whose value is a function of three quantities that are linear in the
# PPI weights, which is what makes the vectorized objective possible. See
# `PreparedCCIScorer` for the derivation.
LINEAR_CCI_SCORES = ('bray_curtis', 'jaccard', 'count', 'icellnet')

# Scores that are not bounded between 0 and 1, and whose distance matrix is
# therefore computed with the regularized formula in `InteractionSpace`.
UNBOUNDED_CCI_SCORES = ('count', 'icellnet')


def _check_if_pygad() -> ModuleType:
    try:
        import pygad

    except Exception:
        raise ImportError('pygad is not installed. Please install it with: '
                          'pip install pygad'
                          )
    return pygad


class PreparedCCIScorer:
    '''
    Precomputes the expression-dependent part of the CCI scores, so that scoring
    the same cells again under a different set of PPI weights becomes a matrix
    multiplication instead of a rebuild of the interaction space.

    Every CCI score in `cell2cell.core.cci_scores` is a function of three
    quantities, each of which is a weighted sum over the PPIs and therefore
    **linear in the PPI weights** `w`:

    .. code-block:: text

        N(i, j)  = sum_k w_k * A_ki * B_kj      (the ligand-receptor product)
        SA(i)    = sum_k w_k * A_ki^2
        SB(j)    = sum_k w_k * B_kj^2

        bray_curtis = 2 * N / (SA + SB)
        jaccard     =     N / (SA + SB - N)
        icellnet    =     N
        count       = sum_k [w_k * A_ki * B_kj != 0]

    Since `A` and `B` depend only on the expression data, they can be computed
    once and reused for any number of weight vectors. Scoring a whole population
    of weight vectors is then a single matrix product.

    Parameters
    ----------
    interaction_space : cell2cell.core.interaction_space.InteractionSpace
        A built interaction space. Only its expression-derived matrices are used;
        it is not modified.

    cci_score : str, default=None
        CCI score to compute. If None, the one the interaction space was built
        with is used. Must be one of 'bray_curtis', 'jaccard', 'count' or
        'icellnet'.

    max_memory_mb : float, default=512
        Budget for the precomputed ligand-receptor outer products, which are what
        make the batched path possible. If the array would exceed this, only the
        per-vector path is prepared, which is still much faster than rebuilding
        the interaction space but does not gain from batching.

    Attributes
    ----------
    A, B : numpy.ndarray
        Weighted ligand and receptor expression, of shape (PPIs, cells).

    cell_names : list
        Cell names, in the order they have in the interaction space. Rows and
        columns of every returned matrix follow this order.

    batched : boolean
        Whether the precomputed outer products fitted in `max_memory_mb`.
    '''

    def __init__(self, interaction_space, cci_score=None, max_memory_mb=512):
        if cci_score is None:
            cci_score = interaction_space.cci_score
        if cci_score not in LINEAR_CCI_SCORES:
            raise NotImplementedError(
                "'{}' is not supported by the vectorized scorer. Use one of {}, or "
                "pass fast=False to fall back to the reference implementation."
                .format(cci_score, list(LINEAR_CCI_SCORES)))

        self.cci_score = cci_score
        self.cci_type = interaction_space.cci_type
        self.cell_names = list(interaction_space.interaction_elements['cell_names'])

        cells = interaction_space.interaction_elements['cells']
        self.A = np.column_stack([cells[c].weighted_ppi['A'].values for c in self.cell_names])
        self.B = np.column_stack([cells[c].weighted_ppi['B'].values for c in self.cell_names])

        # `nansum` in the scalar scores treats missing values as zero. Doing the
        # substitution once here reproduces that without a NaN-aware reduction in
        # the inner loop.
        self._has_nans = bool(np.isnan(self.A).any() or np.isnan(self.B).any())
        self.A = np.nan_to_num(self.A)
        self.B = np.nan_to_num(self.B)

        self.n_ppi, self.n_cells = self.A.shape
        self._A2 = self.A * self.A
        self._B2 = self.B * self.B

        # Outer product of every ligand-receptor pair, flattened over the two cell
        # axes so the contraction over PPIs is a plain matrix product.
        outer_mb = self.n_ppi * self.n_cells * self.n_cells * 8 / 1e6
        self.batched = outer_mb <= max_memory_mb
        if self.batched:
            outer = self.A[:, :, None] * self.B[:, None, :]
            if self.cci_score == 'count':
                # 'count' counts the non-zero products rather than adding them up,
                # which is linear in a binary weight vector but not in a general one.
                self._P = (outer != 0).astype(float).reshape(self.n_ppi, -1)
            else:
                self._P = outer.reshape(self.n_ppi, -1)
        else:
            self._P = None

    def _terms(self, W):
        '''Computes N, SA and SB for a stack of weight vectors W of shape (n, PPIs).'''
        SA = W @ self._A2
        SB = W @ self._B2
        if self.batched:
            N = (W @ self._P).reshape(-1, self.n_cells, self.n_cells)
        else:
            N = np.stack([(self.A * w[:, None]).T @ self.B for w in W])
        return N, SA, SB

    def _combine(self, N, SA, SB):
        '''Applies the score-specific formula. Shapes: N (n, C, C), SA/SB (n, C).'''
        if self.cci_score == 'icellnet' or self.cci_score == 'count':
            return N

        denominator = SA[:, :, None] + SB[:, None, :]
        if self.cci_score == 'jaccard':
            denominator = denominator - N

        with np.errstate(divide='ignore', invalid='ignore'):
            if self.cci_score == 'bray_curtis':
                scores = np.divide(2.0 * N, denominator)
            else:
                scores = np.divide(N, denominator)
        # The scalar implementations return 0.0 when the denominator is zero
        scores[denominator == 0.0] = 0.0
        return scores

    def score_batch(self, W):
        '''
        Computes the CCI matrix for each of several PPI weight vectors.

        Parameters
        ----------
        W : numpy.ndarray
            Weights, of shape (n, PPIs). One row per weight vector.

        Returns
        -------
        scores : numpy.ndarray
            CCI matrices, of shape (n, cells, cells). Rows and columns follow
            `self.cell_names`. For an undirected interaction space the matrices
            are symmetrized the same way `compute_pairwise_cci_scores` does, by
            mirroring the upper triangle.
        '''
        W = np.atleast_2d(np.asarray(W, dtype=float))
        if W.shape[1] != self.n_ppi:
            raise ValueError('Expected weight vectors of length {}, got {}'
                             .format(self.n_ppi, W.shape[1]))

        if self.batched and self.cci_score == 'count' and not np.isin(W, (0.0, 1.0)).all():
            raise ValueError("The 'count' score is only vectorized for binary weights, "
                             "because it counts non-zero products rather than adding them.")

        scores = self._combine(*self._terms(W))

        if self.cci_type == 'undirected':
            # `generate_pairs` yields the upper triangle plus the diagonal, and the
            # scoring loop mirrors each value. The lower triangle of the directed
            # result is therefore never used.
            upper = np.triu(scores)
            scores = upper + np.triu(scores, k=1).transpose(0, 2, 1)
        return scores

    def score(self, ppi_score):
        '''
        Computes the CCI matrix for a single PPI weight vector.

        Parameters
        ----------
        ppi_score : array-like
            Weights, one per PPI.

        Returns
        -------
        cci_matrix : pandas.DataFrame
            CCI scores, with cells as rows and columns.
        '''
        scores = self.score_batch(np.asarray(ppi_score, dtype=float)[None, :])[0]
        return pd.DataFrame(scores, index=self.cell_names, columns=self.cell_names)

    def distance_batch(self, W):
        '''
        Computes the distance matrix for each of several PPI weight vectors,
        reproducing what `InteractionSpace.compute_pairwise_cci_scores` derives.

        Bounded scores use `1 - score`; the unbounded ones ('count', 'icellnet')
        use the regularized `1 - score / (score + mean)`, where the mean is taken
        over the whole CCI matrix of that weight vector. The diagonal is zeroed.

        Parameters
        ----------
        W : numpy.ndarray
            Weights, of shape (n, PPIs).

        Returns
        -------
        distances : numpy.ndarray
            Distance matrices, of shape (n, cells, cells).
        '''
        scores = self.score_batch(W)
        if self.cci_score in UNBOUNDED_CCI_SCORES:
            means = np.nanmean(scores, axis=(1, 2))[:, None, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                distances = 1.0 - np.divide(scores, scores + means)
        else:
            distances = 1.0 - scores

        idx = np.arange(self.n_cells)
        distances[:, idx, idx] = 0.0
        return distances


def _reference_distance_matrix(interaction_space, ppi_score, cells):
    '''Distance matrix through the unmodified `InteractionSpace` code path.'''
    interaction_space.ppi_data['score'] = np.asarray(ppi_score, dtype=float)
    interaction_space.interaction_elements['ppi_score'] = interaction_space.ppi_data['score'].values
    interaction_space.compute_pairwise_cci_scores(use_ppi_score=True, verbose=False)
    return interaction_space.distance_matrix.loc[cells, cells]


def _bidirectional_index(ppi_data, interaction_columns=('A', 'B'), verbose=False):
    '''
    Maps every row of the bidirectional PPI table back to the row of `ppi_data`
    it came from.

    `bidirectional_ppi_for_cci` duplicates every interaction with its partners
    swapped and then drops duplicates, which collapses self-interactions back to a
    single copy. Rather than reimplementing that, this runs it once on a table
    whose score column holds each row's position, and reads the positions back.

    Returns
    -------
    source : numpy.ndarray
        For each row of the bidirectional table, the index of the row of
        `ppi_data` it originates from.
    '''
    probe = ppi_data.copy()
    probe['score'] = np.arange(len(probe), dtype=float)
    bi_probe = bidirectional_ppi_for_cci(ppi_data=probe,
                                         interaction_columns=interaction_columns,
                                         verbose=verbose)
    source = bi_probe['score'].values.astype(int)

    # `drop_duplicates` acts on (A, B, score), so if the table still contains a pair
    # and its reciprocal, two rows that differ only by score stop being duplicates and
    # the bidirectional table changes length with the weights. The mapping is then not
    # well defined -- and neither is assigning that column to a fixed interaction space.
    # The interaction space is built from the all-ones table, so that one has to
    # match too, not just an arbitrary binary vector.
    rng = np.random.default_rng(0)
    probes = [np.ones(len(ppi_data)),
              (np.arange(len(ppi_data)) % 2).astype(float),
              rng.integers(0, 2, size=len(ppi_data)).astype(float)]
    lengths = {len(bidirectional_ppi_for_cci(ppi_data=ppi_data.assign(score=p),
                                             interaction_columns=interaction_columns,
                                             verbose=verbose))
               for p in probes}
    if lengths != {len(source)}:
        raise ValueError(
            'The number of bidirectional interactions depends on the weights, so a '
            'ligand-receptor pair cannot be mapped onto a fixed set of rows. This '
            'happens when `ppi_data` holds a pair and its reciprocal as separate rows, '
            'or the exact same pair more than once. Deduplicate it first -- '
            'cell2cell.preprocessing.remove_ppi_bidirectionality() followed by '
            'drop_duplicates() on the interaction columns.')
    return source


def _correlation(distance_vector, reference_vector, method='spearman'):
    if method == 'spearman':
        corr = scipy.stats.spearmanr(distance_vector, reference_vector)[0]
    elif method == 'pearson':
        corr = scipy.stats.pearsonr(distance_vector, reference_vector)[0]
    else:
        raise ValueError("`method` must be either 'spearman' or 'pearson'")
    return abs(np.nan_to_num(corr))


def lr_selection_frequency(selection_masks):
    '''
    Fraction of independent genetic-algorithm executions that selected each pair.

    Parameters
    ----------
    selection_masks : array-like
        Binary matrix of shape (executions, LR pairs). One row per independent run
        of the genetic algorithm, holding the 0/1 mask it converged to.

    Returns
    -------
    frequency : numpy.ndarray
        Value in [0, 1] per ligand-receptor pair.
    '''
    masks = np.atleast_2d(np.asarray(selection_masks, dtype=float))
    return np.nansum(masks, axis=0) / masks.shape[0]


def lr_cooccurrence(selection_masks, labels=None):
    '''
    Co-occurrence of ligand-receptor pairs across independent genetic-algorithm runs.

    Two pairs co-occur when the same run selected both. The value reported is the
    Jaccard index of their selection patterns -- the number of runs that selected
    both, divided by the number that selected either -- so a pair that is chosen
    rarely can still co-occur strongly with another it is always chosen alongside.

    Parameters
    ----------
    selection_masks : array-like
        Binary matrix of shape (executions, LR pairs), one row per independent run.

    labels : list, default=None
        Names for the ligand-receptor pairs, used as the index and columns of the
        result. If None, positional integers are used.

    Returns
    -------
    cooccurrence : pandas.DataFrame
        Symmetric matrix of Jaccard indexes, with ones on the diagonal for pairs
        selected at least once and zeros for pairs never selected.
    '''
    masks = np.atleast_2d(np.asarray(selection_masks)).astype(bool)
    intersection = (masks.astype(float).T @ masks.astype(float))
    counts = masks.sum(axis=0)
    union = counts[:, None] + counts[None, :] - intersection

    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence = np.divide(intersection, union)
    # A pair of LRs that no run ever selected has an empty union
    cooccurrence[union == 0] = 0.0

    if labels is None:
        labels = list(range(masks.shape[1]))
    return pd.DataFrame(cooccurrence, index=labels, columns=labels)


def consensus_from_cooccurrence(cooccurrence, n_clusters=2, method='ward',
                                select='cooccurrence', frequency=None, min_frequency=0.0):
    '''
    Picks the group of ligand-receptor pairs that keep being selected together.

    Clusters the co-occurrence matrix and returns one cluster: the tight group of
    pairs that are chosen alongside each other across independent runs of the
    genetic algorithm. This is how the published selection for *C. elegans* was
    produced.

    Parameters
    ----------
    cooccurrence : pandas.DataFrame
        Square co-occurrence matrix, as returned by `lr_cooccurrence`. Pairs that no
        run ever selected (an all-zero row and column) are dropped first.

    n_clusters : int, default=2
        Number of clusters to cut the dendrogram into.

    method : str, default='ward'
        Linkage method.

    frequency : array-like, default=None
        Fraction of runs that selected each pair, aligned with `cooccurrence`. Only
        needed for `min_frequency`.

    min_frequency : float, default=0.0
        Drop pairs selected in fewer than this fraction of runs before clustering.
        Two pairs selected once, in the same run, have a Jaccard index of 1.0 even
        though a single run is no evidence that they belong together, and with few
        runs a cluster of such pairs can outscore the genuinely reproducible one.
        This removes them, but it is not a substitute for running the search enough
        times: **the clustering needs on the order of 30 or more runs to be stable**,
        and below that the frequency route is the more reliable of the two. Requires
        `frequency`.

    select : str, default='cooccurrence'
        Which cluster to return.

        - 'cooccurrence' : the one with the highest mean co-occurrence among its own
            members, i.e. the most consistently co-selected group. This is the
            intent of the analysis and does not depend on cluster sizes.
        - 'smallest' : the one with fewest members. This is literally what the
            reference notebook did, and on that data it is also the highest-
            co-occurrence one; it is kept for exact reproducibility.

    Returns
    -------
    selected : list
        Labels of the pairs in the chosen cluster.

    clusters : dict
        Every cluster, keyed by its scipy cluster id, so the others can be inspected.

    scores : dict
        Mean intra-cluster co-occurrence per cluster id, which is what `select`
        ranks on.
    '''
    import scipy.cluster.hierarchy as hc
    from sklearn.metrics import pairwise_distances

    keep = (cooccurrence != 0).any(axis=0)
    if min_frequency > 0.0:
        if frequency is None:
            raise ValueError('`frequency` is required when `min_frequency` is set')
        frequency = np.asarray(frequency, dtype=float)
        if len(frequency) != cooccurrence.shape[0]:
            raise ValueError('`frequency` must have one value per pair in `cooccurrence`')
        keep = keep & (frequency >= min_frequency)
    data = cooccurrence.loc[keep, keep]
    if data.shape[0] < n_clusters:
        raise ValueError('Only {} pairs passed the filters, which is fewer than the {} '
                         'clusters requested. Lower `min_frequency`, or run more '
                         'executions.'.format(data.shape[0], n_clusters))

    # Distances between co-occurrence profiles, then Ward on those. `hc.linkage`
    # reads a square array as observations x features, so it clusters the rows of
    # the distance matrix -- which is what the reference analysis did.
    distances = pairwise_distances(data.values)
    with warnings.catch_warnings():
        # scipy notices that a square hollow matrix was passed where it usually takes
        # a condensed one, and warns. That is what is meant here: the rows of the
        # distance matrix are the observations, matching the reference analysis.
        warnings.simplefilter('ignore', hc.ClusterWarning)
        linkage = hc.linkage(distances, method=method, optimal_ordering=True)

    clusters = get_clusters_from_linkage(linkage, n_clusters, criterion='maxclust',
                                         labels=list(data.index))

    # Mean co-occurrence between distinct members of each cluster
    scores = {}
    for key, members in clusters.items():
        if len(members) < 2:
            scores[key] = 0.0
            continue
        block = data.loc[members, members].values
        off_diagonal = block[~np.eye(len(members), dtype=bool)]
        scores[key] = float(np.nanmean(off_diagonal))

    if select == 'cooccurrence':
        chosen = max(scores, key=lambda k: scores[k])
    elif select == 'smallest':
        chosen = min(clusters, key=lambda k: len(clusters[k]))
    else:
        raise ValueError("`select` must be either 'cooccurrence' or 'smallest'")
    return clusters[chosen], clusters, scores


def consensus_from_frequency(frequency, percentile=90):
    '''
    Keeps the ligand-receptor pairs selected most often across independent runs.

    The simpler alternative to `consensus_from_cooccurrence`: rather than asking
    which pairs are chosen *together*, it asks which are chosen *often*. Cheaper to
    reason about, but it cannot separate two groups of pairs that are each
    self-consistent yet rarely co-selected.

    Parameters
    ----------
    frequency : array-like
        Fraction of runs that selected each pair, from `lr_selection_frequency`.

    percentile : float, default=90
        Percentile of the frequency distribution used as the cutoff. Pairs strictly
        above it are kept. The reference analysis used the 90th percentile.

    Returns
    -------
    mask : numpy.ndarray
        Boolean, True for the pairs that are kept.

    threshold : float
        The cutoff value.
    '''
    values = np.asarray(frequency, dtype=float)
    threshold = float(np.percentile(values, percentile))
    return values > threshold, threshold



def _optimize_once(rnaseq_data, ppi_data, reference_distances, cutoff_setup, analysis_setup,
                    included_cells=None, population_size=200, generations=200, runs=None,
                    inc_percentage=0.025, max_runs=100, correlation='spearman',
                    mutation_probability=0.05, keep_elitism=1, random_state=None,
                    interaction_columns=('A', 'B'), complex_sep=None, complex_agg_method='min',
                    fast=True, validate_fast=True, max_memory_mb=512, deduplicate=True,
                    verbose=False):
    '''
    Selects the subset of ligand-receptor pairs whose cell-cell interaction scores
    best reproduce a reference distance between cells, using a genetic algorithm.

    Each individual is a binary vector with one entry per ligand-receptor pair,
    indicating whether it is included. The objective function is the absolute
    Spearman correlation between the resulting CCI distance matrix and
    `reference_distances`, as in Armingol et al. (2022) on the whole body of *C. elegans*.

    The search is repeated in successive runs: each run keeps only the pairs that
    the previous one selected, so the set shrinks until the objective stops
    improving by at least `inc_percentage`.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression matrix, with genes as rows and cells as columns.

    ppi_data : pandas.DataFrame
        List of ligand-receptor pairs. A 'score' column is added if missing.

    reference_distances : pandas.DataFrame
        Square, symmetric matrix of reference distances between cells, for example
        physical distances. Rows and columns are cell names.

    cutoff_setup : dict
        Cutoff setup, as in `cell2cell.analysis.initialize_interaction_space`.

    analysis_setup : dict
        Analysis setup with the keys 'communication_score', 'cci_score' and
        'cci_type', as in `cell2cell.analysis.initialize_interaction_space`.
        `cci_type` must be 'undirected', since the objective compares the
        condensed form of a symmetric distance matrix.

    included_cells : list, default=None
        Cells to consider. If None, the cells present in both `rnaseq_data` and
        `reference_distances` are used.

    population_size : int, default=200
        Number of individuals per generation.

    generations : int, default=200
        Number of generations per run.

    runs : int, default=None
        Number of runs. If None, runs continue until the objective improves by
        less than `inc_percentage` with respect to the previous run, or until
        `max_runs` is reached.

    inc_percentage : float, default=0.025
        Minimum relative improvement of the objective for another run to start.
        Only used when `runs` is None.

    max_runs : int, default=100
        Upper bound on the number of runs when `runs` is None.

    correlation : str, default='spearman'
        Correlation between the CCI distances and the reference distances, either
        'spearman' or 'pearson'.

    mutation_probability : float, default=0.05
        Probability of flipping each gene, equivalent to the flip mutator of the
        original implementation.

    keep_elitism : int, default=1
        Number of best individuals carried over to the next generation.

    random_state : int, default=None
        Seed for reproducibility.

    interaction_columns : tuple, default=('A', 'B')
        Columns of `ppi_data` holding the ligands and the receptors.

    complex_sep : str, default=None
        Separator of the subunits of a protein complex, if any.

    complex_agg_method : str, default='min'
        Method to aggregate the expression of the subunits of a complex.

    fast : boolean, default=True
        Whether to evaluate the objective with `PreparedCCIScorer`, which is
        equivalent but vectorized. If False, every individual is evaluated by
        rebuilding the CCI scores through `InteractionSpace`, exactly as the
        original implementation did.

    validate_fast : boolean, default=True
        Whether to check the vectorized objective against the reference one on a
        few random individuals before starting. Cheap, and it catches the case
        where the two paths would disagree.

    max_memory_mb : float, default=512
        Memory budget for the precomputed ligand-receptor outer products.

    deduplicate : boolean, default=True
        Whether to collapse each interaction and its reciprocal into a single row
        with `remove_ppi_bidirectionality` before the search. This is required for
        a pair to map onto a fixed set of bidirectional rows: `bidirectional_ppi_for_cci`
        drops duplicates on (A, B, score), so if both directions are present as
        separate rows, the number of bidirectional rows depends on the candidate
        solution. Pairs loaded with `cell2cell.io.load_ppi` are already deduplicated
        by `preprocess_ppi_data`, so this is a no-op for them; it matters when the
        table comes from somewhere else. Turn it off only if the input is already
        deduplicated -- the ambiguous case is rejected either way.
        Note that the returned masks are then indexed against the deduplicated
        table, which is also what 'best_ppi_data' contains.

    verbose : boolean, default=False
        Whether to print the progress of each run.

    Returns
    -------
    results : dict
        Dictionary with one entry per run, keyed 'run1', 'run2', ..., each with:

        - 'obj_fn' : the objective function of the best individual.
        - 'ppi_data' : list of 0/1, one per row of the original `ppi_data`,
          indicating the pairs selected in that run.
        - 'drop_fraction' : fraction of the pairs available to that run that were
          dropped.
        - 'n_selected' : number of pairs selected.

        The dictionary also holds 'best_run', 'best_obj_fn' and 'best_ppi_data',
        the last being a copy of `ppi_data` restricted to the selected pairs.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> results = c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq,
    ...                                          ppi_data=lr_pairs,
    ...                                          reference_distances=physical_distances,
    ...                                          cutoff_setup={'type': 'constant_value',
    ...                                                        'parameter': 10},
    ...                                          analysis_setup={'communication_score': 'expression_thresholding',
    ...                                                          'cci_score': 'bray_curtis',
    ...                                                          'cci_type': 'undirected'},
    ...                                          random_state=888)
    >>> selected = results['best_ppi_data']
    '''
    pygad = _check_if_pygad()

    if analysis_setup['cci_type'] != 'undirected':
        raise NotImplementedError("Only 'undirected' interactions are supported, because the "
                                "objective compares condensed symmetric distance matrices.")

    reference_distances = _as_symmetric(reference_distances)

    # Cells shared by the expression data and the reference distances
    if included_cells is None:
        included_cells = sorted(set(rnaseq_data.columns) & set(reference_distances.columns))
    included_cells = list(included_cells)
    if len(included_cells) < 3:
        raise ValueError('At least three cells are needed to correlate distances')

    reference_vector = scipy.spatial.distance.squareform(
        np.asarray(reference_distances.loc[included_cells, included_cells].values, dtype=float),
        checks=False)

    if deduplicate:
        # Required for the pair-to-bidirectional-row mapping to be well defined; see
        # `_bidirectional_index`. Also what the reference analysis effectively had,
        # since its LR list held each interaction once.
        ppi_data = remove_ppi_bidirectionality(ppi_data=ppi_data,
                                             interaction_columns=interaction_columns,
                                             verbose=verbose)
        ppi_data = ppi_data.drop_duplicates(subset=list(interaction_columns))
        ppi_data = ppi_data.reset_index(drop=True)

    theta_ppi_data = ppi_data.copy()
    if 'score' not in theta_ppi_data.columns:
        theta_ppi_data = theta_ppi_data.assign(score=1.0)

    prot_a, prot_b = interaction_columns
    results = dict()
    run = 1
    previous_obj = None

    while True:
        if runs is None:
            if run > max_runs:
                break
        elif run > runs:
            break

        # Each run searches only among the pairs the previous run kept
        theta_ppi_data = theta_ppi_data.loc[theta_ppi_data['score'] == 1].reset_index(drop=True)
        n_ppi = len(theta_ppi_data)
        if n_ppi == 0:
            break

        bi_ppi_data = bidirectional_ppi_for_cci(ppi_data=theta_ppi_data,
                                              interaction_columns=interaction_columns,
                                              verbose=verbose)
        interaction_space = InteractionSpace(rnaseq_data=rnaseq_data[included_cells],
                                           ppi_data=bi_ppi_data,
                                           gene_cutoffs=cutoff_setup,
                                           communication_score=analysis_setup['communication_score'],
                                           cci_score=analysis_setup['cci_score'],
                                           cci_type=analysis_setup['cci_type'],
                                           complex_sep=complex_sep,
                                           complex_agg_method=complex_agg_method,
                                           interaction_columns=interaction_columns,
                                           verbose=verbose)

        # Position in `theta_ppi_data` that each bidirectional row comes from, so a
        # candidate solution can be expanded to the weights the scorer expects
        source = _bidirectional_index(theta_ppi_data,
                                    interaction_columns=interaction_columns,
                                    verbose=verbose)

        space_cells = list(interaction_space.interaction_elements['cell_names'])
        take = [space_cells.index(c) for c in included_cells]

        def reference_objective(theta):
            weights = np.asarray(theta, dtype=float)[source]
            distances = _reference_distance_matrix(interaction_space, weights, included_cells)
            vector = scipy.spatial.distance.squareform(np.asarray(distances.values, dtype=float),
                                                     checks=False)
            return _correlation(vector, reference_vector, method=correlation)

        use_fast = fast
        if use_fast:
            try:
                scorer = PreparedCCIScorer(interaction_space,
                                         cci_score=analysis_setup['cci_score'],
                                         max_memory_mb=max_memory_mb)
            except NotImplementedError:
                if verbose:
                    print('Falling back to the reference objective for this CCI score')
                use_fast = False

        if use_fast and scorer._has_nans and analysis_setup['cci_score'] == 'count':
            # `count` treats a NaN product as active, which the substitution above
            # does not reproduce. Only this score is affected.
            use_fast = False

        if use_fast:
            def batch_objective(THETA):
                W = np.asarray(THETA, dtype=float)[:, source]
                distances = scorer.distance_batch(W)[:, take][:, :, take]
                out = np.empty(len(W))
                for n, d in enumerate(distances):
                    vector = scipy.spatial.distance.squareform(d, checks=False)
                    out[n] = _correlation(vector, reference_vector, method=correlation)
                return out

            if validate_fast:
                rng = np.random.default_rng(random_state)
                probes = rng.integers(0, 2, size=(2, n_ppi)).astype(float)
                fast_values = batch_objective(probes)
                ref_values = np.array([reference_objective(p) for p in probes])
                if not np.allclose(fast_values, ref_values, rtol=1e-9, atol=1e-9):
                    raise RuntimeError(
                      'The vectorized objective disagrees with the reference one '
                      '({} vs {}). Please report this, and use fast=False meanwhile.'
                      .format(fast_values, ref_values))

            def fitness_func(ga_instance, solution, solution_idx):
                return float(batch_objective(np.atleast_2d(solution))[0])
        else:
            def fitness_func(ga_instance, solution, solution_idx):
                return float(reference_objective(solution))

        ga = pygad.GA(num_generations=generations,
                    num_parents_mating=max(2, population_size // 2),
                    fitness_func=fitness_func,
                    sol_per_pop=population_size,
                    num_genes=n_ppi,
                    gene_type=int,
                    init_range_low=0,
                    init_range_high=2,
                    gene_space=[0, 1],
                    parent_selection_type='tournament',
                    keep_elitism=keep_elitism,
                    mutation_type='random',
                    mutation_probability=mutation_probability,
                    random_seed=random_state if random_state is None else random_state + run,
                    suppress_warnings=True,
                    )
        if use_fast:
            # Evaluate the whole generation with one matrix product
            ga.fitness_batch_size = population_size

            def batch_fitness(ga_instance, solutions, solutions_indices):
                return list(batch_objective(np.atleast_2d(solutions)))

            ga.fitness_func = batch_fitness

        ga.run()

        best_solution, best_fitness, _ = ga.best_solution()
        best = np.asarray(best_solution, dtype=int)

        theta_ppi_data['score'] = best.astype(float)
        drop_fraction = 1.0 - best.sum() / len(best)

        # Map the selection back onto the rows of the original ppi_data
        selected = theta_ppi_data.loc[theta_ppi_data['score'] == 1, [prot_a, prot_b]]
        selected_pairs = set(map(tuple, selected.values))
        mask = [1 if tuple(row) in selected_pairs else 0
                for row in ppi_data[[prot_a, prot_b]].values]

        results['run{}'.format(run)] = {'obj_fn': float(best_fitness),
                                      'ppi_data': mask,
                                      'drop_fraction': float(drop_fraction),
                                      'n_selected': int(best.sum()),
                                      }
        if verbose:
            print('Run {}: objective {:.4f}, {} of {} pairs kept'
                  .format(run, best_fitness, int(best.sum()), len(best)))

        if runs is None and previous_obj is not None:
            if (best_fitness - previous_obj) / previous_obj < inc_percentage:
                run += 1
                break
        previous_obj = best_fitness
        run += 1

    if not results:
        raise RuntimeError('The genetic algorithm produced no results')

    best_key = max(results, key=lambda k: results[k]['obj_fn'])
    best_mask = np.asarray(results[best_key]['ppi_data'], dtype=bool)
    results['best_run'] = best_key
    results['best_obj_fn'] = results[best_key]['obj_fn']
    results['best_ppi_data'] = ppi_data.loc[best_mask].reset_index(drop=True)
    return results

def optimize_lr_pairs(rnaseq_data, ppi_data, reference_distances, cutoff_setup, analysis_setup,
                      executions=1, random_state=None, consensus_method='cooccurrence',
                      n_clusters=2, cluster_selection='cooccurrence', min_frequency=0.0,
                      frequency_percentile=90, verbose=False, **kwargs):
    '''
    Selects ligand-receptor pairs whose cell-cell interaction scores best reproduce a
    reference distance between cells, using a genetic algorithm.

    A genetic algorithm converges to a *local* optimum, so a single execution is not
    conclusive: different seeds settle on different, largely overlapping sets of
    pairs. With `executions > 1` the search is repeated independently and the results
    are integrated the way the reference analysis did -- by how often each pair is
    selected, and by which pairs are selected *together* -- which separates the
    reproducible core from the noise of any one execution.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression matrix, with genes as rows and cells as columns.

    ppi_data : pandas.DataFrame
        List of ligand-receptor pairs. A 'score' column is added if missing.

    reference_distances : pandas.DataFrame
        Square, symmetric matrix of reference distances between cells, for example
        physical distances. Rows and columns are cell names.

    cutoff_setup : dict
        Cutoff setup, as in `cell2cell.analysis.initialize_interaction_space`.

    analysis_setup : dict
        Analysis setup with the keys 'communication_score', 'cci_score' and
        'cci_type'. `cci_type` must be 'undirected'.

    executions : int, default=1
        Number of independent runs of the genetic algorithm. Each uses a different
        seed derived from `random_state`. With more than one, the consensus outputs
        described below are added to the result. **Around 30 or more is needed for
        the co-occurrence clustering to be stable**; the reference analysis used
        about a hundred. The frequency route tolerates fewer.

    random_state : int, default=None
        Seed. Execution *i* uses `random_state + i`, so the whole set is reproducible.

    consensus_method : str, default='cooccurrence'
        How to integrate the executions when there is more than one.

        - 'cooccurrence' : cluster the pairs by how often they are selected
            *together* and keep one cluster. This is what the reference analysis
            did, and it is able to tell apart groups of pairs that are each
            self-consistent.
        - 'frequency' : keep the pairs selected most often, above
            `frequency_percentile`. Simpler, and blind to which pairs go together.

    n_clusters : int, default=2
        Number of clusters to cut the co-occurrence dendrogram into.

    cluster_selection : str, default='cooccurrence'
        Which cluster to keep: 'cooccurrence' for the one whose members co-occur
        most with each other, or 'smallest' for the fewest members, which is
        literally what the reference notebook did. See `consensus_from_cooccurrence`.

    min_frequency : float, default=0.0
        Drop pairs selected in fewer than this fraction of executions before building
        the co-occurrence clusters, removing pairs that co-occur perfectly only
        because they were each chosen once, in the same execution. Note the
        co-occurrence route needs roughly 30 or more executions to be stable; with
        fewer, prefer `consensus_method='frequency'`. See
        `consensus_from_cooccurrence`.

    frequency_percentile : float, default=90
        Cutoff percentile when `consensus_method='frequency'`.

    verbose : boolean, default=False
        Whether to print the progress of each execution.

    **kwargs
        Passed to each individual search: `population_size`, `generations`, `runs`,
        `inc_percentage`, `max_runs`, `correlation`, `mutation_probability`,
        `keep_elitism`, `included_cells`, `interaction_columns`, `complex_sep`,
        `complex_agg_method`, `fast`, `validate_fast`, `max_memory_mb` and
        `deduplicate`. See `_optimize_once` for their meaning.

    Returns
    -------
    results : dict
        With a single execution, the result of that search: one entry per run
        ('run1', 'run2', ...) with 'obj_fn', 'ppi_data' (a 0/1 mask over the rows of
        `ppi_data`), 'drop_fraction' and 'n_selected', plus 'best_run',
        'best_obj_fn' and 'best_ppi_data'.

        With several executions, the same keys report the single best execution, and
        these are added:

        - 'executions' : the full result of each execution, keyed 'execution1', ...
        - 'selection_masks' : binary array of shape (executions, LR pairs), the mask
          each execution converged to.
        - 'selection_frequency' : dataframe of the pairs with the fraction of
          executions that selected each one.
        - 'cooccurrence' : Jaccard co-occurrence between pairs across executions.
        - 'consensus_ppi_data' : the consensus selection -- with the default method,
          the pairs of the co-occurrence cluster whose members are most consistently
          chosen together. **This is the recommended output.**
        - 'consensus_clusters' : every cluster, so the others can be inspected.
        - 'consensus_cluster_scores' : mean intra-cluster co-occurrence per cluster.

        With `consensus_method='frequency'`, 'consensus_threshold' holds the
        frequency cutoff instead of the cluster keys.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> results = c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq,
    ...                                          ppi_data=lr_pairs,
    ...                                          reference_distances=physical_distances,
    ...                                          cutoff_setup={'type': 'constant_value',
    ...                                                        'parameter': 10},
    ...                                          analysis_setup={'communication_score': 'expression_thresholding',
    ...                                                          'cci_score': 'bray_curtis',
    ...                                                          'cci_type': 'undirected'},
    ...                                          executions=20, random_state=888)
    >>> results['consensus_ppi_data']
    '''
    if executions < 1:
        raise ValueError('`executions` must be at least 1')

    common = dict(rnaseq_data=rnaseq_data, ppi_data=ppi_data,
                  reference_distances=reference_distances, cutoff_setup=cutoff_setup,
                  analysis_setup=analysis_setup, verbose=verbose, **kwargs)

    if executions == 1:
        return _optimize_once(random_state=random_state, **common)

    interaction_columns = kwargs.get('interaction_columns', ('A', 'B'))
    prot_a, prot_b = interaction_columns

    # The pool the masks are indexed against, matching what each execution searches
    pool = ppi_data
    if kwargs.get('deduplicate', True):
        pool = remove_ppi_bidirectionality(ppi_data=ppi_data,
                                           interaction_columns=interaction_columns,
                                           verbose=False)
        pool = pool.drop_duplicates(subset=list(interaction_columns)).reset_index(drop=True)

    all_executions, masks = {}, []
    for i in range(executions):
        seed = None if random_state is None else random_state + i
        result = _optimize_once(random_state=seed, **common)
        all_executions['execution{}'.format(i + 1)] = result
        # The pairs that execution converged to, i.e. its last run
        last = max((k for k in result if k.startswith('run')),
                   key=lambda k: int(k[3:]))
        masks.append(result[last]['ppi_data'])
        if verbose:
            print('Execution {}: best objective {:.4f}, {} pairs'
                  .format(i + 1, result['best_obj_fn'], sum(result[last]['ppi_data'])))

    masks = np.asarray(masks, dtype=int)
    labels = ['{}^{}'.format(a, b) for a, b in pool[[prot_a, prot_b]].values]

    frequency = pd.DataFrame({prot_a: pool[prot_a].values, prot_b: pool[prot_b].values,
                              'frequency': lr_selection_frequency(masks)})
    cooccurrence = lr_cooccurrence(masks, labels=labels)

    best_key = max(all_executions, key=lambda k: all_executions[k]['best_obj_fn'])
    results = dict(all_executions[best_key])
    results['executions'] = all_executions
    results['best_execution'] = best_key
    results['selection_masks'] = masks
    results['selection_frequency'] = frequency.sort_values('frequency', ascending=False)
    results['cooccurrence'] = cooccurrence

    if consensus_method == 'frequency':
        mask, threshold = consensus_from_frequency(frequency['frequency'].values,
                                                   percentile=frequency_percentile)
        results['consensus_ppi_data'] = pool.loc[mask].reset_index(drop=True)
        results['consensus_threshold'] = threshold
        if verbose:
            print('Frequency consensus: {} pairs above {:.3f}'
                  .format(int(mask.sum()), threshold))
    elif consensus_method == 'cooccurrence':
        try:
            selected, clusters, scores = consensus_from_cooccurrence(
                cooccurrence, n_clusters=n_clusters, select=cluster_selection,
                frequency=frequency['frequency'].values, min_frequency=min_frequency)
            chosen = set(selected)
            results['consensus_ppi_data'] = pool.loc[[l in chosen for l in labels]].reset_index(drop=True)
            results['consensus_clusters'] = clusters
            results['consensus_cluster_scores'] = scores
            if verbose:
                print('Co-occurrence consensus: {} pairs, cluster sizes {}, mean co-occurrence {}'
                      .format(len(chosen), {k: len(v) for k, v in clusters.items()},
                              {k: round(v, 3) for k, v in scores.items()}))
        except ValueError as error:
            if verbose:
                print('No consensus could be built: {}'.format(error))
            results['consensus_ppi_data'] = None
            results['consensus_clusters'] = None
            results['consensus_cluster_scores'] = None
    else:
        raise ValueError("`consensus_method` must be either 'cooccurrence' or 'frequency'")
    return results


def _as_symmetric(matrix):
    '''
    Validates a reference distance matrix and makes it exactly symmetric.

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
