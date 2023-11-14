# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances


def celltype_pair_distance(df1, df2, method='min', distance='euclidean'):
    '''
    Calculates the distance between two sets of data points (single cell coordinates)
    represented by df1 and df2. It supports two distance metrics: Euclidean and Manhattan
    distances. The method parameter allows you to specify how the distances between the
    two sets are aggregated.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first set of single cell coordinates.

    df1 : pandas.DataFrame
        The second set of single cell coordinates.

    method : str, default='min'
        The aggregation method for the calculated distances. It can be one of 'min',
        'max', or 'mean'.

    distance : str, default='euclidean'
        The distance metric to use. It can be 'euclidean' or 'manhattan'.

    Returns
    -------
    agg_dist : numpy.float
        The aggregated distance between the two sets of data points based on the specified
        method and distance metric.
    '''
    if distance == 'euclidean':
        distances = euclidean_distances(df1, df2)
    elif distance == 'manhattan':
        distances = manhattan_distances(df1, df2)
    else:
        raise NotImplementedError("{} distance is not implemented.".format(distance.capitalize()))

    if method == 'min':
        agg_dist = np.nanmin(distances)
    elif method == 'max':
        agg_dist = np.nanmax(distances)
    elif method == 'mean':
        agg_dist = np.nanmean(distances)
    else:
        raise NotImplementedError('Method {} is not implemented.'.format(method))
    return agg_dist


def pairwise_celltype_distances(df, group_col, coord_cols=['X', 'Y'],
                                method='min', distance='euclidean', pairs=None):
    '''
    Calculates pairwise distances between groups of single cells. It computes an
    aggregate distance between all possible combinations of groups.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe where each row is a single cell, and there are columns containing
        spatial coordinates and cell group.

    group_col : str
        The name of the column that defines the groups for which distances are calculated.

    coord_cols : list, default=None
        The list of column names that represent the coordinates of the single cells.

    pairs : list
        A list of specific group pairs for which distances should be calculated.
        If not provided, all possible combinations of group pairs will be considered.

    Returns
    -------
    distances : pandas.DataFrame
        The pairwise distances between groups based on the specified group column.
        In this dataframe rows and columns are the cell groups used to compute distances.
    '''
    # TODO: Adapt code below to receive AnnData or MuData objects
    # df_ = pd.DataFrame(adata.obsm['spatial'], index=adata.obs_names, columns=['X', 'Y'])
    # df = adata.obs[[group_col]]
    df_ = df[coord_cols]
    groups = df[group_col].unique()
    distances = pd.DataFrame(np.zeros((len(groups), len(groups))),
                             index=groups,
                             columns=groups)

    if pairs is None:
        pairs = list(itertools.combinations(groups, 2))

    for pair in pairs:
        dist = celltype_pair_distance(df_.loc[df[group_col] == pair[0]], df_.loc[df[group_col] == pair[1]],
                                      method=method,
                                      distance=distance
                                      )
        distances.loc[pair[0], pair[1]] = dist
        distances.loc[pair[1], pair[0]] = dist
    return distances