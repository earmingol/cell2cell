import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from cell2cell.clustering import compute_linkage
from cell2cell.clustering.cluster_interactions import compute_distance
from cell2cell.plotting.aesthetics import get_colors_from_labels


def clustermap_ccc(interaction_space, metadata=None, sample_col='#SampleID', group_col='Groups',
                   meta_cmap='gist_rainbow', colors=None, cell_labels=('SENDER-CELL','RECEIVER-CELL'),
                   metric='jaccard', method='ward', optimal_leaf=True, excluded_cells=None, title='',
                   only_used_lr=True, cbar_title='Presence', cbar_fontsize=12, row_fontsize=8, col_fontsize=8,
                   filename=None, **kwargs):
    '''Generates a clustermap (heatmap + dendrograms from a hierarchical
    clustering) based on CCC scores for each LR pair in every cell-cell pair.

    Parameters
    ----------
    interaction_space : cell2cell.core.interaction_space.InteractionSpace
        Interaction space that contains all a distance matrix after running the
        the method compute_pairwise_communication_scores. Alternatively, this
        object can be a numpy-array or a pandas DataFrame. Also, a
        SingleCellInteractions or a BulkInteractions object after running
        the method compute_pairwise_communication_scores.

    metadata : pandas.Dataframe, default=None
        Metadata associated with the cells, cell types or samples in the
        matrix containing CCC scores. If None, cells will not be colored
        by major groups.

    sample_col : str, default='#SampleID'
        Column in the metadata for the cells, cell types or samples
        in the matrix containing CCC scores.

    group_col : str, default='Groups'
        Column in the metadata containing the major groups of cells, cell types
        or samples in the matrix with CCC scores.

    meta_cmap : str, default='gist_rainbow'
        Name of the color palette for coloring the major groups of cells.

    colors : dict, default=None
        Dictionary containing tuples in the RGBA format for indicating colors
        of major groups of cells. If colors is specified, meta_cmap will be
        ignored.

    cell_labels : tuple, default=('SENDER-CELL','RECEIVER-CELL')
        A tuple containing the labels for indicating the group colors of
        sender and receiver cells if metadata or colors are provided.

    metric : str, default='jaccard'
        The distance metric to use. The distance function can be 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.

    method : str, default='ward'
        Clustering method for computing a linkage as in
        scipy.cluster.hierarchy.linkage

    optimal_leaf : boolean, default=True
        Whether sorting the leaf of the dendrograms to have a minimal distance
        between successive leaves. For more information, see
        scipy.cluster.hierarchy.optimal_leaf_ordering

    excluded_cells : list, default=None
        List containing cell names that are present in the interaction_space
        object but that will be excluded from this plot.

    title : str, default=''
        Title of the clustermap.

    only_used_lr : boolean, default=True
        Whether displaying or not only LR pairs that were used at least by
        one pair of cells. If True, those LR pairs that were not used will
        not be displayed.

    cbar_title : str, default='CCI score'
        Title for the colorbar, depending on the score employed.

    cbar_fontsize : int, default=12
        Font size for the colorbar title as well as labels for axes X and Y.

    row_fontsize : int, default=8
        Font size for the rows in the clustermap (ligand-receptor pairs).

    col_fontsize : int, default=8
        Font size for the columns in the clustermap (sender-receiver cell pairs).

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    **kwargs : dict
        Dictionary containing arguments for the seaborn.clustermap function.

    Returns
    -------
    fig : seaborn.matrix.ClusterGrid
        A seaborn ClusterGrid instance.
    '''
    if hasattr(interaction_space, 'interaction_elements'):
        print('Interaction space detected as an InteractionSpace class')
        if 'communication_matrix' not in interaction_space.interaction_elements.keys():
            raise ValueError('First run the method compute_pairwise_communication_scores() in your interaction' + \
                             ' object to generate a communication matrix.')
        else:
            df_ = interaction_space.interaction_elements['communication_matrix'].copy()
    elif (type(interaction_space) is np.ndarray) or (type(interaction_space) is pd.core.frame.DataFrame):
        print('Interaction space detected as a communication matrix')
        df_ = interaction_space
    elif hasattr(interaction_space, 'interaction_space'):
        print('Interaction space detected as a Interactions class')
        if 'communication_matrix' not in interaction_space.interaction_space.interaction_elements.keys():
            raise ValueError('First run the method compute_pairwise_communication_scores() in your interaction' + \
                             ' object to generate a communication matrix.')
        else:
            df_ = interaction_space.interaction_space.interaction_elements['communication_matrix'].copy()
    else:
        raise ValueError('First run the method compute_pairwise_communication_scores() in your interaction' + \
                         ' object to generate a communication matrix.')

    if excluded_cells is not None:
        included_cells = []
        for cells in df_.columns:
            include = True
            for excluded_cell in excluded_cells:
                if excluded_cell in cells:
                    include = False
            if include:
                included_cells.append(cells)
    else:
        included_cells = list(df_.columns)

    df_ = df_[included_cells]
    df_ = df_.dropna(how='all', axis=0)
    df_ = df_.dropna(how='all', axis=1)
    df_ = df_.fillna(0)
    if only_used_lr:
        df_ = df_[(df_.T != 0).any()]

    # Clustering
    dm_rows = compute_distance(df_, axis=0, metric=metric)
    row_linkage = compute_linkage(dm_rows, method=method, optimal_ordering=optimal_leaf)

    dm_cols = compute_distance(df_, axis=1, metric=metric)
    col_linkage = compute_linkage(dm_cols, method=method, optimal_ordering=optimal_leaf)

    # Colors
    if metadata is not None:
        metadata2 = metadata.reset_index()
        meta_ = metadata2.set_index(sample_col)
        if excluded_cells is not None:
            meta_ = meta_.loc[~meta_.index.isin(excluded_cells)]
        labels = meta_[group_col].values.tolist()

        if colors is None:
            colors = get_colors_from_labels(labels, cmap=meta_cmap)
        else:
            assert all(elem in colors.keys() for elem in set(labels))

        col_colors_L = pd.DataFrame(included_cells)[0].apply(lambda x: colors[metadata2.loc[metadata2[sample_col] == x.split(';')[0],
                                                                                           group_col].values[0]])
        col_colors_L.index = included_cells
        col_colors_L.name = cell_labels[0]

        col_colors_R = pd.DataFrame(included_cells)[0].apply(lambda x: colors[metadata2.loc[metadata2[sample_col] == x.split(';')[1],
                                                                                           group_col].values[0]])
        col_colors_R.index = included_cells
        col_colors_R.name = cell_labels[1]

        # Clustermap
        fig = sns.clustermap(df_,
                             cmap=sns.dark_palette('red'), #plt.get_cmap('YlGnBu_r'),
                             col_linkage=col_linkage,
                             row_linkage=row_linkage,
                             col_colors=[col_colors_L,
                                         col_colors_R],
                             **kwargs
                             )

        fig.ax_heatmap.set_yticklabels(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, ha='left')
        fig.ax_heatmap.set_xticklabels(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

        fig.ax_col_colors.set_yticks(np.arange(0.5, 2., step=1))
        fig.ax_col_colors.set_yticklabels(list(cell_labels), fontsize=row_fontsize)
        fig.ax_col_colors.yaxis.tick_right()
        plt.setp(fig.ax_col_colors.get_yticklabels(), rotation=0, visible=True)


    else:
        fig = sns.clustermap(df_,
                             cmap=sns.dark_palette('red'),
                             col_linkage=col_linkage,
                             row_linkage=row_linkage,
                             **kwargs
                             )

    # Title
    if len(title) > 0:
        fig.ax_col_dendrogram.set_title(title, fontsize=24)

    # Color bar label
    cbar = fig.ax_heatmap.collections[0].colorbar
    cbar.ax.set_ylabel(cbar_title, fontsize=cbar_fontsize)
    cbar.ax.yaxis.set_label_position("left")

    # Change tick labels
    xlabels = [' --> '.join(i.get_text().split(';')) \
               for i in fig.ax_heatmap.xaxis.get_majorticklabels()]
    ylabels = [' --> '.join(i.get_text().replace('(', '').replace(')', '').replace("'", "").split(', ')) \
               for i in fig.ax_heatmap.yaxis.get_majorticklabels()]

    fig.ax_heatmap.set_xticklabels(xlabels,
                                   fontsize=col_fontsize,
                                   rotation=90,
                                   rotation_mode='anchor',
                                   va='center',
                                   ha='right')
    fig.ax_heatmap.set_yticklabels(ylabels)

    # Save Figure
    if filename is not None:
        plt.savefig(filename,
                    dpi=300,
                    bbox_inches='tight')
    return fig