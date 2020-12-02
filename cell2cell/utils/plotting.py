# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

import scipy.cluster.hierarchy as hc
import scipy.spatial as sp

import skbio


def get_colors_from_labels(labels, cmap='gist_rainbow', factor=1):
    assert factor >= 1

    colors = dict.fromkeys(labels, ())

    factor = int(factor)
    NUM_COLORS = factor * len(colors)
    cm = plt.get_cmap(cmap)

    for i, label in enumerate(colors.keys()):
        colors[label] = cm((1 + ((factor-1)/factor)) * i / NUM_COLORS)
    return colors


# TODO: @Erick, modify rotation with anchor. Also, create separe function for clustermap without clustering
def clustermap_cci(interaction_space, method='ward', optimal_leaf=True, metadata=None, sample_col='#SampleID',
                   group_col='Groups', meta_cmap='gist_rainbow', colors=None, excluded_cells=None, title='',
                   cbar_title='CCI score', cbar_fontsize='12', filename=None, **kwargs):
    if hasattr(interaction_space, 'distance_matrix'):
        print('Interaction space detected as an InteractionSpace class')
        distance_matrix = interaction_space.distance_matrix
        space_type = 'class'
    elif (type(interaction_space) is np.ndarray) or (type(interaction_space) is pd.core.frame.DataFrame):
        print('Interaction space detected as a distance matrix')
        distance_matrix = interaction_space
        space_type = 'matrix'
    else:
        raise ValueError('First run InteractionSpace.compute_pairwise_interactions() to generate a distance matrix.')

    # Drop excluded cells
    if excluded_cells is not None:
        df = distance_matrix.loc[~distance_matrix.index.isin(excluded_cells),
                                 ~distance_matrix.columns.isin(excluded_cells)]
    else:
        df = distance_matrix

    # Check symmetry to run linkage separately
    symmetric = (df.values.transpose() == df.values).all()
    if symmetric:
        # Remove diagonal
        df_diag = np.diag(df.values)
        np.fill_diagonal(df.values, 0.0)

        # Compute linkage
        D = sp.distance.squareform(df)
        if 'col_cluster' in kwargs.keys():
            kwargs['row_cluster'] = kwargs['col_cluster']
            if kwargs['col_cluster']:
                linkage = hc.linkage(D, method=method, optimal_ordering=optimal_leaf)
            else:
                linkage = None
        else:
            linkage = hc.linkage(D, method=method, optimal_ordering=optimal_leaf)

        df = df + np.diag(df_diag)
        # Plot hierarchical clustering
        hier = sns.clustermap(df,
                              col_linkage=linkage,
                              row_linkage=linkage,
                              **kwargs
                              )
        plt.close()

        # Triangular matrix
        order_map = dict()
        if linkage is None:
            #ind_order = range(hier.data2d.shape[0])
            mask = np.ones((df.shape[0], df.shape[1]))
            for i in range(mask.shape[0]):
                for j in range(i, mask.shape[1]):
                    mask[i, j] = 0
        else:
            ind_order = hier.dendrogram_col.reordered_ind
            mask = np.zeros((df.shape[0], df.shape[1]))
            for i, ind in enumerate(ind_order):
                order_map[i] = ind
                filter_list = [order_map[j] for j in range(i)]
                mask[ind, filter_list] = 1 #hier.dendrogram_col.reordered_ind[:i]

        kwargs_ = kwargs.copy()
        # This one does exclude only diagonal
        #exclude_diag = df.values[~np.eye(df.values.shape[0], dtype=bool)].reshape(df.values.shape[0], -1)
        #if 'vmin' not in list(kwargs_.keys()):
        #    vmin = np.nanmin(exclude_diag)
        #    kwargs_['vmin'] = vmin
        #if 'vmax' not in list(kwargs_.keys()):
        #    vmax = np.nanmax(exclude_diag)
        #    kwargs_['vmax'] = vmax
    else:
        linkage = None
        mask = None
        kwargs_ = kwargs.copy()

    # PLOT CCI MATRIX
    if space_type == 'class':
        df = interaction_space.interaction_elements['cci_matrix']
    else:
        df = distance_matrix

    if excluded_cells is not None:
        df = df.loc[~df.index.isin(excluded_cells),
                    ~df.columns.isin(excluded_cells)]

    # Colors
    if metadata is not None:
        meta_ = metadata.set_index(sample_col)
        if excluded_cells is not None:
            meta_ = meta_.loc[~meta_.index.isin(excluded_cells)]

        assert (set(meta_.index) & set(df.columns)) == set(df.columns), "Metadata is not provided for all cells in interaction space"

        labels = meta_[group_col].values.tolist()
        if colors is None:
            colors = get_colors_from_labels(labels, cmap=meta_cmap)
        else:
            assert all(elem in colors.keys() for elem in set(labels)), "Colors are not provided for all groups in metadata"

        col_colors = pd.DataFrame(labels)[0].map(colors)
        col_colors.index = meta_.index
        col_colors.name = group_col.capitalize()

        if not symmetric:
            row_colors = col_colors
        else:
            row_colors = None

        # Plot hierarchical clustering (triangular)
        hier = sns.clustermap(df,
                              col_linkage=linkage,
                              row_linkage=linkage,
                              mask=mask,
                              col_colors=col_colors,
                              row_colors=row_colors,
                              **kwargs_
                              )
    else:
        hier = sns.clustermap(df,
                              col_linkage=linkage,
                              row_linkage=linkage,
                              mask=mask,
                              **kwargs_
                              )

    # Apply offset transform to all xticklabels.
    if symmetric:
        hier.ax_row_dendrogram.set_visible(False)
        hier.ax_heatmap.tick_params(bottom=False)  # Hide xtick line

        labels_x = hier.ax_heatmap.xaxis.get_majorticklabels()
        labels_y = hier.ax_heatmap.yaxis.get_majorticklabels()

        label_x1 = labels_x[0]
        label_y1 = labels_y[0]

        pos_x0 = label_x1.get_transform().transform((0., 0.))
        xlims = hier.ax_heatmap.get_xlim()
        xrange = abs(xlims[0] - xlims[1])
        x_cell_width = xrange / len(df.columns)

        pos_y0 = label_y1.get_transform().transform((0., 0.))
        pos_y1 = label_y1.get_transform().transform(label_y1.get_position())
        ylims = hier.ax_heatmap.get_ylim()
        yrange = abs(ylims[0] - ylims[1])
        y_cell_width = yrange / len(df.index)

        dpi_x = hier.fig.dpi_scale_trans.to_values()[0]
        dpi_y = hier.fig.dpi_scale_trans.to_values()[3]

        for i, label in enumerate(labels_x):
            if label.get_position()[0] % x_cell_width == 0:
                pos_x1 = label.get_transform().transform((aux_scaler_x, aux_scaler_x))
                scaler_x = abs(pos_x1 - pos_x0)
            else:
                pos_x1 = label.get_transform().transform((x_cell_width, x_cell_width))
                scaler_x = abs(pos_x1 - pos_x0)

            aux_scaler_x = label.get_position()[0] - (label.get_position()[0] // x_cell_width) * x_cell_width
            aux_scaler_y = label.get_position()[1] - (label.get_position()[1] // y_cell_width) * y_cell_width

            new_pos_x = label.get_position()[0] - aux_scaler_x
            new_pos_y = new_pos_x * yrange / xrange - yrange + y_cell_width

            new_pos_y = label_y1.get_transform().transform((0, new_pos_y + aux_scaler_y)) - pos_y0
            dx = -0.5 * scaler_x[0] / dpi_x
            dy = new_pos_y[1] / dpi_y
            offset = mlp.transforms.ScaledTranslation(dx, dy, hier.fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)

    if symmetric:
        rot = 45
        ha = 'right'
    else:
        rot = 90
        ha = 'center'
    hier.ax_heatmap.set_xticklabels(hier.ax_heatmap.xaxis.get_majorticklabels(), rotation=rot, ha=ha)#, fontsize=9.5)
    hier.ax_heatmap.set_yticklabels(hier.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, ha='left')  # , fontsize=9.5)

    # Title
    if len(title) > 0:
        hier.ax_col_dendrogram.set_title(title, fontsize=16)

    # Color bar label
    cbar = hier.ax_heatmap.collections[0].colorbar
    cbar.ax.set_ylabel(cbar_title, fontsize=cbar_fontsize)
    cbar.ax.yaxis.set_label_position("left")

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return hier


def pcoa_biplot(interaction_space, metadata, sample_col='#SampleID', group_col='Groups', meta_cmap='gist_rainbow',
                title='', axis_size=14, legend_size=12, figsize=(6, 5), view_angles=(30, 135), filename=None,
                colors=None, excluded_cells=None):

    if hasattr(interaction_space, 'distance_matrix'):
        print('Interaction space detected as an InteractionSpace class')
        distance_matrix = interaction_space.distance_matrix
    elif (type(interaction_space) is np.ndarray) or (type(interaction_space) is pd.core.frame.DataFrame):
        print('Interaction space detected as a distance matrix')
        distance_matrix = interaction_space
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

    # Biplot
    fig = plt.figure(figsize=figsize)
    #ax = fig.add_subplot(111, projection='3d')

    ax = Axes3D(fig)

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
            ax.scatter(ordination.samples.loc[cells, 'PC1'],
                       ordination.samples.loc[cells, 'PC2'],
                       ordination.samples.loc[cells, 'PC3'],
                       color=colors[cell_type],
                       s=50,
                       edgecolors='k',
                       label=cell_type)
        else:
            ax.scatter(ordination.samples.loc[cells, 'PC1'],
                       ordination.samples.loc[cells, 'PC2'],
                       ordination.samples.loc[cells, 'PC3'],
                       s=50,
                       edgecolors='k',
                       label=cell_type)

    # Plot texts
    ax.set_xlabel('PC1 ({}%)'.format(np.round(ordination.proportion_explained['PC1'] * 100), 2), fontsize=axis_size)
    ax.set_ylabel('PC2 ({}%)'.format(np.round(ordination.proportion_explained['PC2'] * 100), 2), fontsize=axis_size)
    ax.set_zlabel('PC3 ({}%)'.format(np.round(ordination.proportion_explained['PC3'] * 100), 2), fontsize=axis_size)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(view_angles[0], view_angles[1])
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
              ncol=2, fancybox=True, shadow=True, fontsize=legend_size)
    plt.title(title, fontsize=16)

    distskbio = skbio.DistanceMatrix(df, ids=df.index)

    # Save plot
    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')

    results = {'fig' : fig, 'axes' : ax, 'ordination' : ordination, 'distance_matrix' : distskbio}
    return results


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
    dm_rows = sp.distance.squareform(sp.distance.pdist(df_.values, metric=metric))
    row_linkage = hc.linkage(dm_rows, method=method, optimal_ordering=optimal_leaf)

    dm_cols = sp.distance.squareform(sp.distance.pdist(df_.values.T, metric=metric))
    col_linkage = hc.linkage(dm_cols, method=method, optimal_ordering=optimal_leaf)

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