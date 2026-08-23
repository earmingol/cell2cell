# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
from natsort import natsorted
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
        'max', 'mean', or 'median'.

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
    elif method == 'median':
        agg_dist = np.nanmedian(distances)
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

def get_spatial_coordinates(adata, spatial_key='spatial', coord_names=None):
    '''
    Extracts the spatial coordinates of an AnnData object as a dataframe.

    Parameters
    ----------
    adata : anndata.AnnData
        Object containing the spatial coordinates of each single cell.

    spatial_key : str, default='spatial'
        Key in `adata.obsm` where the coordinates are stored. Objects written by
        different tools use different keys (e.g. 'spatial', 'X_spatial',
        'X_umap'), so it can be changed here.

    coord_names : list, default=None
        Names to give to the coordinate columns. If None, they are named 'X', 'Y'
        and 'Z' for the first three dimensions, and 'Dim4', 'Dim5', ... beyond
        that, so the result works with the `coord_cols` parameter of the other
        functions in this module.

    Returns
    -------
    coordinates : pandas.DataFrame
        Coordinates of each single cell. Rows are the observation names of
        `adata`, in the same order, and columns are the dimensions.
    '''
    if spatial_key not in adata.obsm.keys():
        raise KeyError("'{}' is not in adata.obsm. Available keys are: {}"
                       .format(spatial_key, list(adata.obsm.keys())))

    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2:
        raise ValueError('The coordinates in adata.obsm[\'{}\'] must be two-dimensional'
                         .format(spatial_key))

    if coord_names is None:
        default = ['X', 'Y', 'Z']
        coord_names = [default[i] if i < len(default) else 'Dim{}'.format(i + 1)
                       for i in range(coords.shape[1])]
    elif len(coord_names) != coords.shape[1]:
        raise ValueError('`coord_names` must have one name per dimension ({})'
                         .format(coords.shape[1]))

    return pd.DataFrame(coords, index=adata.obs_names, columns=coord_names)


def celltype_centroids(adata, group_col, spatial_key='spatial', coord_names=None,
                       method='mean'):
    '''
    Computes the centroid of each cell type from the coordinates of its single cells.

    Parameters
    ----------
    adata : anndata.AnnData or pandas.DataFrame
        Either an AnnData object with coordinates in `adata.obsm[spatial_key]` and
        the cell-type annotation in `adata.obs[group_col]`, or a dataframe with one
        row per single cell containing both the coordinates and the annotation.

    group_col : str
        Column with the cell-type annotation. Taken from `adata.obs` for an AnnData
        object, and from the dataframe itself otherwise.

    spatial_key : str, default='spatial'
        Key in `adata.obsm` where the coordinates are stored. Ignored when a
        dataframe is passed.

    coord_names : list, default=None
        Names of the coordinate columns. For an AnnData object they name the
        extracted dimensions; for a dataframe they select which columns to use. If
        None, an AnnData is named 'X', 'Y', 'Z', ... and for a dataframe every
        column other than `group_col` is used.

    method : str, default='mean'
        How to summarize the coordinates of the single cells of a cell type. It can
        be 'mean' (the centroid proper) or 'median' (the component-wise median,
        which is robust to cells scattered far from the rest of their type).

    Returns
    -------
    centroids : pandas.DataFrame
        One row per cell type and one column per dimension. Cell types are
        naturally sorted, so 'CT-2' comes before 'CT-10'.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> adata = c2c.datasets.generate_toy_spatial_adata()
    >>> centroids = c2c.spatial.celltype_centroids(adata, group_col='cell_type')
    '''
    coords, groups = _coordinates_and_groups(adata, group_col, spatial_key, coord_names)

    if method == 'mean':
        centroids = coords.groupby(groups, observed=True).mean()
    elif method == 'median':
        centroids = coords.groupby(groups, observed=True).median()
    else:
        raise NotImplementedError("Method {} is not implemented. Use 'mean' or 'median'."
                                  .format(method))
    centroids.index.name = group_col
    return centroids.loc[natsorted(centroids.index)]


def celltype_centroid_distances(adata, group_col, spatial_key='spatial', coord_names=None,
                                centroid_method='mean', distance='euclidean'):
    '''
    Computes the distances between the centroids of every pair of cell types.

    This summarizes each cell type by one point before measuring distances, so its
    cost does not depend on how many single cells each type contains. That makes it
    the option to use on large datasets, where the all-versus-all single-cell
    distances of `pairwise_celltype_distances` become prohibitive.

    Parameters
    ----------
    adata : anndata.AnnData or pandas.DataFrame
        Object or dataframe containing the coordinates and the cell-type annotation.

    group_col : str
        Column with the cell-type annotation.

    spatial_key : str, default='spatial'
        Key in `adata.obsm` where the coordinates are stored.

    coord_names : list, default=None
        Names of the coordinate columns.

    centroid_method : str, default='mean'
        How to summarize the coordinates of each cell type, 'mean' or 'median'.

    distance : str, default='euclidean'
        The distance metric to use. It can be 'euclidean' or 'manhattan'.

    Returns
    -------
    distances : pandas.DataFrame
        Symmetric matrix with a zero diagonal, where rows and columns are the cell
        types, naturally sorted.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> adata = c2c.datasets.generate_toy_spatial_adata()
    >>> distances = c2c.spatial.celltype_centroid_distances(adata, group_col='cell_type')
    '''
    centroids = celltype_centroids(adata, group_col, spatial_key=spatial_key,
                                   coord_names=coord_names, method=centroid_method)

    if distance == 'euclidean':
        matrix = euclidean_distances(centroids.values, centroids.values)
    elif distance == 'manhattan':
        matrix = manhattan_distances(centroids.values, centroids.values)
    else:
        raise NotImplementedError("{} distance is not implemented.".format(distance.capitalize()))

    # Forced rather than assumed, so the result always satisfies `check_symmetry`
    # and `squareform`, whatever rounding the metric introduced
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return pd.DataFrame(matrix, index=centroids.index, columns=centroids.index)


def celltype_distances(adata, group_col, spatial_key='spatial', coord_names=None,
                       method='centroid', distance='euclidean', centroid_method='mean',
                       pairs=None, verbose=False):
    '''
    Computes a distance between every pair of cell types from the coordinates of
    their single cells.

    Single entry point for the two ways of summarizing the distance between two
    cell types: aggregating the distances between all of their single cells
    ('min', 'max', 'mean', 'median'), or measuring between their centroids
    ('centroid').

    Parameters
    ----------
    adata : anndata.AnnData or pandas.DataFrame
        Object or dataframe containing the coordinates and the cell-type annotation.

    group_col : str
        Column with the cell-type annotation.

    spatial_key : str, default='spatial'
        Key in `adata.obsm` where the coordinates are stored. Ignored when a
        dataframe is passed.

    coord_names : list, default=None
        Names of the coordinate columns.

    method : str, default='centroid'
        How to summarize the distance between two cell types:

        - 'centroid' : distance between the centroids of the two cell types. Cost
          is independent of the number of single cells, so this is the one to use
          on large datasets.
        - 'min' : smallest distance between any two of their single cells, i.e.
          how close the two types get to each other.
        - 'max' : largest distance between any two of their single cells.
        - 'mean' : average over all pairs of their single cells.
        - 'median' : median over all pairs of their single cells, less sensitive
          to a few distant cells than 'mean'.

        Every option other than 'centroid' evaluates all pairs of single cells of
        the two types, so its cost grows with the product of their sizes.

    distance : str, default='euclidean'
        The distance metric to use. It can be 'euclidean' or 'manhattan'.

    centroid_method : str, default='mean'
        How to summarize the coordinates of each cell type when
        `method='centroid'`, either 'mean' or 'median'.

    pairs : list, default=None
        Specific pairs of cell types to compute. If None, all combinations are
        used. Ignored when `method='centroid'`, which computes all of them at once.

    verbose : boolean, default=False
        Whether to warn when the all-versus-all computation is going to be large.

    Returns
    -------
    distances : pandas.DataFrame
        Symmetric matrix with a zero diagonal, where rows and columns are the cell
        types, naturally sorted.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> adata = c2c.datasets.generate_toy_spatial_adata()
    >>> # Fast, and the sensible default on large data
    >>> distances = c2c.spatial.celltype_distances(adata, group_col='cell_type')
    >>> # How close the two cell types get to each other
    >>> distances = c2c.spatial.celltype_distances(adata, group_col='cell_type',
    ...                                            method='min')
    '''
    if method == 'centroid':
        return celltype_centroid_distances(adata, group_col, spatial_key=spatial_key,
                                           coord_names=coord_names,
                                           centroid_method=centroid_method,
                                           distance=distance)

    coords, groups = _coordinates_and_groups(adata, group_col, spatial_key, coord_names)

    counts = pd.Series(groups).value_counts()
    if verbose:
        worst = int(counts.max()) ** 2
        if worst > 1e8:
            print('Computing all-versus-all distances for up to {:.1e} pairs of single '
                  "cells per cell-type pair. Consider method='centroid'.".format(worst))

    df = coords.copy()
    df[group_col] = groups
    return pairwise_celltype_distances(df, group_col=group_col,
                                       coord_cols=list(coords.columns),
                                       method=method, distance=distance, pairs=pairs)


def _coordinates_and_groups(adata, group_col, spatial_key, coord_names):
    '''
    Normalizes the two accepted inputs into a coordinates dataframe and a list of
    cell-type labels aligned with it.
    '''
    if hasattr(adata, 'obsm'):
        if group_col not in adata.obs.columns:
            raise KeyError("'{}' is not a column of adata.obs".format(group_col))
        coords = get_spatial_coordinates(adata, spatial_key=spatial_key,
                                         coord_names=coord_names)
        groups = np.asarray(adata.obs[group_col].values)
    elif isinstance(adata, pd.DataFrame):
        if group_col not in adata.columns:
            raise KeyError("'{}' is not a column of the dataframe".format(group_col))
        if coord_names is None:
            coord_names = [c for c in adata.columns if c != group_col]
        coords = adata[list(coord_names)]
        groups = np.asarray(adata[group_col].values)
    else:
        raise TypeError('`adata` must be an AnnData object or a pandas DataFrame, got {}'
                        .format(type(adata).__name__))
    return coords, groups
