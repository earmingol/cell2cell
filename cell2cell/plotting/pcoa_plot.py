# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cell2cell.external import pcoa, _check_ordination
from cell2cell.plotting.aesthetics import get_colors_from_labels


def pcoa_3dplot(interaction_space, metadata=None, sample_col='#SampleID', group_col='Groups', pcoa_method='eigh',
                meta_cmap='gist_rainbow', colors=None, excluded_cells=None, title='', axis_fontsize=14, legend_fontsize=12,
                figsize=(6, 5), view_angles=(30, 135), filename=None):
    '''Projects the cells into an Euclidean space (PCoA) given their distances
    based on their CCI scores. Then, plots each cell by their first three
    coordinates in a 3D scatter plot.

    Parameters
    ----------
    interaction_space : cell2cell.core.interaction_space.InteractionSpace
        Interaction space that contains all a distance matrix after running the
        the method compute_pairwise_cci_scores. Alternatively, this object
        can be a numpy-array or a pandas DataFrame. Also, a
        SingleCellInteractions or a BulkInteractions object after running
        the method compute_pairwise_cci_scores.

    metadata : pandas.Dataframe, default=None
        Metadata associated with the cells, cell types or samples in the
        matrix containing CCC scores. If None, cells will not be colored
        by major groups.

    sample_col : str, default='#SampleID'
        Column in the metadata for the cells, cell types or samples
        in the matrix containing CCI scores.

    group_col : str, default='Groups'
        Column in the metadata containing the major groups of cells, cell types
        or samples in the matrix with CCI scores.

    pcoa_method : str, default='eigh'
        Eigendecomposition method to use in performing PCoA.
        By default, uses SciPy's `eigh`, which computes exact
        eigenvectors and eigenvalues for all dimensions. The alternate
        method, `fsvd`, uses faster heuristic eigendecomposition but loses
        accuracy. The magnitude of accuracy lost is dependent on dataset.

    meta_cmap : str, default='gist_rainbow'
        Name of the color palette for coloring the major groups of cells.

    colors : dict, default=None
        Dictionary containing tuples in the RGBA format for indicating colors
        of major groups of cells. If colors is specified, meta_cmap will be
        ignored.

    excluded_cells : list, default=None
        List containing cell names that are present in the interaction_space
        object but that will be excluded from this plot.

    title : str, default=''
        Title of the PCoA 3D plot.

    axis_fontsize : int, default=14
        Size of the font for the labels of each axis (X, Y and Z).

    legend_fontsize : int, default=12
        Size of the font for labels in the legend.

    figsize : tuple, default=(6, 5)
        Size of the figure (width*height), each in inches.

    view_angles : tuple, default=(30, 135)
        Rotation angles of the plot. Set the elevation and
        azimuth of the axes.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    Returns
    -------
    results : dict
        Dictionary that contains:

        - 'fig' : matplotlib.figure.Figure, containing the whole figure
        - 'axes' : matplotlib.axes.Axes, containing the axes of the 3D plot
        - 'ordination' : Ordination or projection obtained from the PCoA
        - 'distance_matrix' : Distance matrix used to perform the PCoA (usually in
            interaction_space.distance_matrix
    '''
    if hasattr(interaction_space, 'distance_matrix'):
        print('Interaction space detected as an InteractionSpace class')
        distance_matrix = interaction_space.distance_matrix
    elif (type(interaction_space) is np.ndarray) or (type(interaction_space) is pd.core.frame.DataFrame):
        print('Interaction space detected as a distance matrix')
        distance_matrix = interaction_space
    elif hasattr(interaction_space, 'interaction_space'):
        print('Interaction space detected as a Interactions class')
        if not hasattr(interaction_space.interaction_space, 'distance_matrix'):
            raise ValueError('First run the method compute_pairwise_interactions() in your interaction' + \
                             ' object to generate a distance matrix.')
        else:
            distance_matrix = interaction_space.interaction_space.distance_matrix
    else:
        raise ValueError('First run the method compute_pairwise_interactions() in your interaction' + \
                         ' object to generate a distance matrix.')

    # Drop excluded cells
    if excluded_cells is not None:
        df = distance_matrix.loc[~distance_matrix.index.isin(excluded_cells),
                                 ~distance_matrix.columns.isin(excluded_cells)]
    else:
        df = distance_matrix

    # PCoA
    ordination = pcoa(df, method=pcoa_method)
    ordination = _check_ordination(ordination)
    ordination['samples'].index = df.index

    # Biplot
    fig = plt.figure(figsize=figsize)
    #ax = fig.add_subplot(111, projection='3d')

    ax = Axes3D(fig)

    if metadata is None:
        metadata = pd.DataFrame()
        metadata[sample_col] = list(distance_matrix.columns)
        metadata[group_col] = list(distance_matrix.columns)

    meta_ = metadata.set_index(sample_col)
    if excluded_cells is not None:
        meta_ = meta_.loc[~meta_.index.isin(excluded_cells)]
    labels = meta_[group_col].values.tolist()

    if colors is None:
        colors = get_colors_from_labels(labels, cmap=meta_cmap)
    else:
        assert all(elem in colors.keys() for elem in set(labels))

    # Plot each data point with respective color
    for i, cell_type in enumerate(sorted(meta_[group_col].unique())):
        cells = list(meta_.loc[meta_[group_col] == cell_type].index)
        if colors is not None:
            ax.scatter(ordination['samples'].loc[cells, 'PC1'],
                       ordination['samples'].loc[cells, 'PC2'],
                       ordination['samples'].loc[cells, 'PC3'],
                       color=colors[cell_type],
                       s=50,
                       edgecolors='k',
                       label=cell_type)
        else:
            ax.scatter(ordination['samples'].loc[cells, 'PC1'],
                       ordination['samples'].loc[cells, 'PC2'],
                       ordination['samples'].loc[cells, 'PC3'],
                       s=50,
                       edgecolors='k',
                       label=cell_type)

    # Plot texts
    ax.set_xlabel('PC1 ({}%)'.format(np.round(ordination['proportion_explained']['PC1'] * 100), 2), fontsize=axis_fontsize)
    ax.set_ylabel('PC2 ({}%)'.format(np.round(ordination['proportion_explained']['PC2'] * 100), 2), fontsize=axis_fontsize)
    ax.set_zlabel('PC3 ({}%)'.format(np.round(ordination['proportion_explained']['PC3'] * 100), 2), fontsize=axis_fontsize)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(view_angles[0], view_angles[1])
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
              ncol=2, fancybox=True, shadow=True, fontsize=legend_fontsize)
    plt.title(title, fontsize=16)

    #distskbio = skbio.DistanceMatrix(df, ids=df.index) # Not using skbio for now

    # Save plot
    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')

    results = {'fig' : fig, 'axes' : ax, 'ordination' : ordination, 'distance_matrix' : df} # df used to be distskbio
    return results