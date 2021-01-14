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
                   only_used_lr=True, cbar_title='Presence', cbar_fontsize=12, row_fontsize=8, filename=None, **kwargs):

    if hasattr(interaction_space, 'interaction_elements'):
        print('Interaction space detected as an InteractionSpace class')
        df_ = interaction_space.interaction_elements['communication_matrix'].copy()
    elif (type(interaction_space) is np.ndarray) or (type(interaction_space) is pd.core.frame.DataFrame):
        print('Interaction space detected as a communication matrix')
        df_ = interaction_space
    else:
        raise ValueError('First run InteractionSpace.compute_pairwise_communication_scores() to generate a communication matrix.')

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

    dm_cols = compute_distance(df_.values.T, axis=1, metric=metric)
    col_linkage = compute_linkage(dm_cols, method=method, optimal_ordering=optimal_leaf)

    # Colors
    if metadata is not None:
        meta_ = metadata.set_index(sample_col)
        if excluded_cells is not None:
            meta_ = meta_.loc[~meta_.index.isin(excluded_cells)]
        labels = meta_[group_col].values.tolist()

        if colors is None:
            colors = get_colors_from_labels(labels, cmap=meta_cmap)
        else:
            assert all(elem in colors.keys() for elem in set(labels))

        col_colors_L = pd.DataFrame(included_cells)[0].apply(lambda x: colors[metadata.loc[metadata[sample_col] == x.split(';')[0],
                                                                                           group_col].values[0]])
        col_colors_L.index = included_cells
        col_colors_L.name = cell_labels[0]

        col_colors_R = pd.DataFrame(included_cells)[0].apply(lambda x: colors[metadata.loc[metadata[sample_col] == x.split(';')[1],
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

    # Save Figure
    if filename is not None:
        plt.savefig(filename,
                    dpi=300,
                    bbox_inches='tight')
    return fig