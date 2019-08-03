# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as hc
import scipy.spatial as sp

import skbio


def hierarchical_clustering(distance_matrix, method='ward', title='', filename=None, **kwargs):
    # Compute linkage
    linkage = hc.linkage(sp.distance.squareform(distance_matrix), method=method)

    # Plot hierarchical clustering
    hier = sns.clustermap(distance_matrix,
                          col_linkage=linkage,
                          row_linkage=linkage,
                          )
    hier.fig.suptitle(title)
    plt.close()

    # Triangular matrix
    mask = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))
    order_map = dict()
    for i, ind in enumerate(hier.dendrogram_col.reordered_ind):
        order_map[i] = ind

        mask[ind, hier.dendrogram_col.reordered_ind[:i]] = 1

    vmax = np.nanmax(distance_matrix.values[distance_matrix.values != 0])
    vmin = np.nanmin(distance_matrix.values[distance_matrix.values != 0])

    # Plot hierarchical clustering (triagular)
    hier = sns.clustermap(distance_matrix,
                          col_linkage=linkage,
                          row_linkage=linkage,
                          vmin=vmin,
                          vmax=vmax,
                          mask=mask,
                          **kwargs
                          )

    hier.ax_row_dendrogram.set_visible(False)
    hier.ax_heatmap.tick_params(bottom=False)  # Hide xtick line

    # Apply offset transform to all x ticklabels.
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
        dx = -0.5 * scaler_x[0] / 72.;
        dy = scaler_y[1] * (yrange - i - .5) / 72.
        offset = mlp.transforms.ScaledTranslation(dx, dy, hier.fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)

    hier.ax_heatmap.set_xticklabels(hier.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9.5)

    hier.fig.suptitle(title)
    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return hier


def pcoa_biplot(distance_matrix, metadata, sample_col='#SampleID', group_col='Groups', colors=None,
                title='', axis_size=14, legend_size=12, figsize=(6, 5), view_angles=(30, 135), filename=None):
    # PCoA
    ordination = skbio.stats.ordination.pcoa(distance_matrix)
    ordination.samples.index = distance_matrix.index

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for i, cell_type in enumerate(sorted(metadata[group_col].unique())):
        cells = metadata.loc[metadata[group_col] == cell_type][sample_col].values
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
    lgd = ax.legend(loc='upper right', bbox_to_anchor=(2.4, 0.71),
                    ncol=2, fancybox=True, shadow=True, fontsize=legend_size)
    plt.title(title)

    distskbio = skbio.DistanceMatrix(distance_matrix, ids=distance_matrix.index)
    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')

    results = {'fig' : fig, 'axes' : ax, 'ordination' : ordination, 'distance_matrix' : distskbio}
    return results