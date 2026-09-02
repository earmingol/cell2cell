# -*- coding: utf-8 -*-

'''Integrating repeated selections of ligand-receptor pairs.

A genetic algorithm converges to a local optimum, so a single run is not
conclusive. These functions integrate several independent runs by how often
each pair is chosen, and by which pairs are chosen together.
'''

from __future__ import absolute_import

import warnings

import numpy as np
import pandas as pd

from cell2cell.clustering.cluster_interactions import get_clusters_from_linkage


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
