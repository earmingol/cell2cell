# -*- coding: utf-8 -*-
import umap

import pandas as pd
import scipy.spatial as sp


def run_umap(rnaseq_data, axis=1, metric='euclidean', min_dist=0.4, n_neighbors=8, random_state=None, **kwargs):
    '''Runs UMAP on a expression matrix.
    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        A dataframe of gene expression values wherein the rows are the genes or
        embeddings of a dimensionality reduction method and columns the cells,
        tissues or samples.

    axis : int, default=0
        An axis of the dataframe (0 across rows, 1 across columns).
        Across rows means that the UMAP is to compare genes, while
        across columns is to compare cells, tissues or samples.

    metric : str, default='euclidean'
        The distance metric to use. The distance function can be 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.

    min_dist: float, default=0.4
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    n_neighbors: float, default=8
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    random_state : int, default=None
        Seed for randomization.

    **kwargs : dict
        Extra arguments for UMAP as defined in umap.UMAP.

    Returns
    -------
    umap_df : pandas.DataFrame
        Dataframe containing the UMAP embeddings for the axis analyzed.
        Contains columns 'umap1 and 'umap2'.
    '''
    # Organize data
    if axis == 0:
        df = rnaseq_data
    elif axis == 1:
        df = rnaseq_data.T
    else:
        raise ValueError("The parameter axis must be either 0 or 1.")

    # Compute distances
    D = sp.distance.pdist(df, metric=metric)
    D_sq = sp.distance.squareform(D)

    # Run UMAP
    model = umap.UMAP(metric="precomputed",
                      min_dist=min_dist,
                      n_neighbors=n_neighbors,
                      random_state=random_state,
                      **kwargs
                      )

    trans_D = model.fit_transform(D_sq)

    # Organize results
    umap_df = pd.DataFrame(trans_D, columns=['umap1', 'umap2'], index=df.index)
    return umap_df