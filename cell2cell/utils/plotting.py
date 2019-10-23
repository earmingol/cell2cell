# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as hc
import scipy.spatial as sp

import skbio


def get_colors_from_labels(labels, cmap='gist_rainbow', factor=1):
    assert factor >= 1

    factor = int(factor)
    NUM_COLORS = factor * len(labels)
    cm = plt.get_cmap(cmap)

    colors = dict()
    for i, label in enumerate(labels):
        colors[label] = cm((1 + ((factor-1)/factor)) * i / NUM_COLORS)
    return colors


def clustermap_cci(interaction_space, method='ward', metadata=None, sample_col='#SampleID', group_col='Groups',
                   meta_cmap='gist_rainbow', title='', filename=None, excluded_cells=None, colors=None, **kwargs):
    if hasattr(interaction_space, 'distance_matrix'):
        distance_matrix = interaction_space.distance_matrix
    else:
        raise ValueError('First run InteractionSpace.compute_pairwise_interactions() to generate a distance matrix.')

    # Drop excluded cells
    if excluded_cells is not None:
        df = distance_matrix.loc[~distance_matrix.index.isin(excluded_cells),
                                 ~distance_matrix.columns.isin(excluded_cells)]
    else:
        df = distance_matrix

    # Compute linkage
    linkage = hc.linkage(sp.distance.squareform(df), method=method)

    # Plot hierarchical clustering
    hier = sns.clustermap(df,
                          col_linkage=linkage,
                          row_linkage=linkage,
                          )
    hier.fig.suptitle(title)
    plt.close()

    # Triangular matrix
    mask = np.zeros((df.shape[0], df.shape[1]))
    order_map = dict()
    for i, ind in enumerate(hier.dendrogram_col.reordered_ind):
        order_map[i] = ind

        mask[ind, hier.dendrogram_col.reordered_ind[:i]] = 1

    # This may not work if we have a perfect interaction between two cells - Intended to avoid diagonal in distance matrix
    # exclude_diag = distance_matrix.values[distance_matrix.values != 0]

    kwargs_ = kwargs.copy()

    # This one does exclude only diagonal
    #exclude_diag = df.values[~np.eye(df.values.shape[0], dtype=bool)].reshape(df.values.shape[0], -1)
    #if 'vmin' not in list(kwargs_.keys()):
    #    vmin = np.nanmin(exclude_diag)
    #    kwargs_['vmin'] = vmin
    #if 'vmax' not in list(kwargs_.keys()):
    #    vmax = np.nanmax(exclude_diag)
    #    kwargs_['vmax'] = vmax

    df = interaction_space.interaction_elements['cci_matrix']
    df = df.loc[~df.index.isin(excluded_cells),
                ~df.columns.isin(excluded_cells)]

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

        col_colors = pd.DataFrame(labels)[0].map(colors)
        col_colors.index = meta_.index
        col_colors.name = group_col.capitalize()

        # Plot hierarchical clustering (triangular)
        hier = sns.clustermap(df,
                              col_linkage=linkage,
                              row_linkage=linkage,
                              mask=mask,
                              col_colors=col_colors,
                              **kwargs_
                              )
    else:
        hier = sns.clustermap(df,
                              col_linkage=linkage,
                              row_linkage=linkage,
                              mask=mask,
                              **kwargs_
                              )

    hier.ax_row_dendrogram.set_visible(False)
    hier.ax_heatmap.tick_params(bottom=False)  # Hide xtick line

    # Apply offset transform to all xticklabels.
    yrange = hier.ax_heatmap.get_ylim()[0] - hier.ax_heatmap.get_ylim()[1]

    label_x1 = hier.ax_heatmap.xaxis.get_majorticklabels()[0]
    label_x2 = hier.ax_heatmap.xaxis.get_majorticklabels()[1]

    scaler_x = abs(label_x1.get_transform().transform(label_x1.get_position()) \
                   - label_x2.get_transform().transform(label_x2.get_position()))

    label_y1 = hier.ax_heatmap.yaxis.get_majorticklabels()[0]
    label_y2 = hier.ax_heatmap.yaxis.get_majorticklabels()[1]

    scaler_y = abs(label_y1.get_transform().transform(label_y1.get_position()) \
                   - label_y2.get_transform().transform(label_y2.get_position()))

    for i, label in enumerate(hier.ax_heatmap.xaxis.get_majorticklabels()):
        # scaler = 19./72. 72 is for inches
        dx = -0.5 * scaler_x[0] / 72.
        dy = scaler_y[1] * (yrange - i - .5) / 72.
        offset = mlp.transforms.ScaledTranslation(dx, dy, hier.fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)

    hier.ax_heatmap.set_xticklabels(hier.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')#, fontsize=9.5)
    hier.ax_heatmap.set_yticklabels(hier.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, ha='left')  # , fontsize=9.5)
    hier.fig.suptitle(title, fontsize=16)

    # Color bar label
    cbar = hier.ax_heatmap.collections[0].colorbar
    cbar.ax.set_ylabel('CCI score')
    cbar.ax.yaxis.set_label_position("left")

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return hier


def pcoa_biplot(interaction_space, metadata, sample_col='#SampleID', group_col='Groups', meta_cmap='gist_rainbow',
                title='', axis_size=14, legend_size=12, figsize=(6, 5), view_angles=(30, 135), filename=None,
                colors=None, excluded_cells=None):

    if hasattr(interaction_space, 'distance_matrix'):
        distance_matrix = interaction_space.distance_matrix
    else:
        raise ValueError('First run InteractionSpace.compute_pairwise_interactions() to generate a distance matrix.')

    # Drop excluded cells
    if excluded_cells is not None:
        df = distance_matrix.loc[~distance_matrix.index.isin(excluded_cells),
                                 ~distance_matrix.columns.isin(excluded_cells)]
    else:
        df = distance_matrix

    # PCoA
    ordination = skbio.stats.ordination.pcoa(df, method='eigh')
    ordination.samples.index = df.index

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    meta_ = metadata.set_index(sample_col)
    if excluded_cells is not None:
        meta_ = meta_.loc[~meta_.index.isin(excluded_cells)]
    labels = meta_[group_col].values.tolist()


    if colors is None:
        colors = get_colors_from_labels(labels, cmap=meta_cmap)
    else:
        assert all(elem in colors.keys() for elem in set(labels))

    for i, cell_type in enumerate(sorted(meta_[group_col].unique())):
        cells = list(meta_.loc[meta_[group_col] == cell_type].index)
        if colors is not None:
            ax.scatter(ordination.samples.loc[cells, 'PC1'],
                       ordination.samples.loc[cells, 'PC2'],
                       ordination.samples.loc[cells, 'PC3'],
                       color=colors[cell_type],
                       s=50, label=cell_type)
        else:
            ax.scatter(ordination.samples.loc[cells, 'PC1'],
                       ordination.samples.loc[cells, 'PC2'],
                       ordination.samples.loc[cells, 'PC3'],
                       s=50, label=cell_type)

    ax.set_xlabel('PC1 ({}%)'.format(np.round(ordination.proportion_explained['PC1'] * 100), 2), fontsize=axis_size)
    ax.set_ylabel('PC2 ({}%)'.format(np.round(ordination.proportion_explained['PC2'] * 100), 2), fontsize=axis_size)
    ax.set_zlabel('PC3 ({}%)'.format(np.round(ordination.proportion_explained['PC3'] * 100), 2), fontsize=axis_size)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(view_angles[0], view_angles[1])
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5),
              ncol=2, fancybox=True, shadow=True, fontsize=legend_size)
    plt.title(title, fontsize=16)

    distskbio = skbio.DistanceMatrix(df, ids=df.index)
    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')

    results = {'fig' : fig, 'axes' : ax, 'ordination' : ordination, 'distance_matrix' : distskbio}
    return results



def clustermap_cell_pairs_vs_ppi(ppi_score_for_cell_pairs, metadata=None, sample_col='#SampleID', group_col='Groups',
                                 meta_cmap='gist_rainbow', colors=None, metric='cosine', method='ward', excluded_cells=None,
                                 title='', filename=None,
                                 **kwargs):
    df_ = ppi_score_for_cell_pairs.copy()

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
    df_ = df_[(df_.T != 0).any()]

    # Clustering
    dm_rows = sp.distance.squareform(sp.distance.pdist(df_.values, metric=metric))
    row_linkage = hc.linkage(dm_rows, method=method)

    dm_cols = sp.distance.squareform(sp.distance.pdist(df_.values.T, metric=metric))
    col_linkage = hc.linkage(dm_cols, method=method)

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
        col_colors_L.name = 'LEFT-CELL'

        col_colors_R = pd.DataFrame(included_cells)[0].apply(lambda x: colors[metadata.loc[metadata[sample_col] == x.split(';')[1],
                                                                                           group_col].values[0]])
        col_colors_R.index = included_cells
        col_colors_R.name = 'RIGHT-CELL'

        # Clustermap
        fig = sns.clustermap(df_,
                             cmap=sns.dark_palette('red'),
                             col_linkage=col_linkage,
                             row_linkage=row_linkage,
                             col_colors=[col_colors_L,
                                         col_colors_R],
                             **kwargs
                             )

        fig.ax_col_colors.set_yticks(np.arange(0.5, 2., step=1))
        fig.ax_col_colors.set_yticklabels(['LEFT-CELL', 'RIGHT-CELL'], fontsize=12)
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
        fig.fig.suptitle(title, fontsize=16)

    # Color bar label
    cbar = fig.ax_heatmap.collections[0].colorbar
    cbar.ax.set_ylabel('Presence')
    cbar.ax.yaxis.set_label_position("left")

    # Save Figure
    if filename is not None:
        plt.savefig(filename,
                    dpi=300,
                    bbox_inches='tight')
    return fig