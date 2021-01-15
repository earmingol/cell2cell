# -*- coding: utf-8 -*-

import types
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from cell2cell.clustering import compute_linkage
from cell2cell.preprocessing.manipulate_dataframes import check_symmetry
from cell2cell.plotting.aesthetics import map_colors_to_metadata


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

    # Check symmetry to get linkage
    symmetric = check_symmetry(df)
    if (not symmetric) & (type(interaction_space) is pd.core.frame.DataFrame):
        assert set(df.index) == set(df.columns), 'The distance matrix does not have the same elements in rows and columns'

    # Obtain info for generating plot
    linkage = _get_distance_matrix_linkages(df=df,
                                            kwargs=kwargs,
                                            method=method,
                                            optimal_ordering=optimal_leaf,
                                            symmetric=symmetric
                                            )

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
        col_colors = map_colors_to_metadata(df=df,
                                            metadata=metadata,
                                            colors=colors,
                                            sample_col=sample_col,
                                            group_col=group_col,
                                            meta_cmap=meta_cmap)

        if not symmetric:
            row_colors = col_colors
        else:
            row_colors = None
    else:
        col_colors = None
        row_colors = None

    # Plot hierarchical clustering (triangular)
    hier = _plot_triangular_clustermap(df=df,
                                       symmetric=symmetric,
                                       linkage=linkage,
                                       col_colors=col_colors,
                                       row_colors=row_colors,
                                       title=title,
                                       cbar_title=cbar_title,
                                       cbar_fontsize=cbar_fontsize,
                                       **kwargs_)

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return hier


def _get_distance_matrix_linkages(df, kwargs, method='ward', optimal_ordering=True, symmetric=None):
    if symmetric is None:
        symmetric = check_symmetry(df)

    if symmetric:
        if 'col_cluster' in kwargs.keys():
            kwargs['row_cluster'] = kwargs['col_cluster']
            if kwargs['col_cluster']:
                linkage = compute_linkage(df, method=method, optimal_ordering=optimal_ordering)
            else:
                linkage = None
        elif 'row_cluster' in kwargs.keys():
            if kwargs['row_cluster']:
                linkage = compute_linkage(df, method=method, optimal_ordering=optimal_ordering)
            else:
                linkage = None
        else:
            linkage = compute_linkage(df, method=method, optimal_ordering=optimal_ordering)
    else:
        linkage = None
    return linkage


def _triangularize_distance_matrix(df, linkage=None, symmetric=None, **kwargs):
    if symmetric is None:
        symmetric = check_symmetry(df)

    # Triangular matrix
    if symmetric:
        order_map = dict()
        if linkage is None:
            mask = np.ones((df.shape[0], df.shape[1]))
            for i in range(mask.shape[0]):
                for j in range(i, mask.shape[1]):
                    mask[i, j] = 0
        else:
            # Plot hierarchical clustering for getting indexes according to linkage
            hier = sns.clustermap(df,
                                  col_linkage=linkage,
                                  row_linkage=linkage,
                                  **kwargs
                                  )
            plt.close()
            ind_order = hier.dendrogram_col.reordered_ind
            mask = np.zeros((df.shape[0], df.shape[1]))
            for i, ind in enumerate(ind_order):
                order_map[i] = ind
                filter_list = [order_map[j] for j in range(i)]
                mask[ind, filter_list] = 1
    else:
        mask = None
    return mask


def _plot_triangular_clustermap(df, symmetric=None, linkage=None, mask=None, col_colors=None, row_colors=None,
                                title='', cbar_title='CCI score', cbar_fontsize='12', **kwargs):
    if symmetric is None:
        symmetric = check_symmetry(df)

    if mask is None:
        mask = _triangularize_distance_matrix(df=df,
                                              linkage=linkage,
                                              symmetric=symmetric,
                                              **kwargs
                                              )

    hier = sns.clustermap(df,
                          col_linkage=linkage,
                          row_linkage=linkage,
                          mask=mask,
                          col_colors=col_colors,
                          row_colors=row_colors,
                          **kwargs
                          )

    hier = _move_xticks_triangular_clustermap(clustermap=hier,
                                              symmetric=symmetric
                                              )

    # Title
    if len(title) > 0:
        hier.ax_col_dendrogram.set_title(title, fontsize=16)

    # Color bar label
    cbar = hier.ax_heatmap.collections[0].colorbar
    cbar.ax.set_ylabel(cbar_title, fontsize=cbar_fontsize)
    cbar.ax.yaxis.set_label_position("left")
    return hier


def _move_xticks_triangular_clustermap(clustermap, symmetric=True):
    if symmetric:
    # Apply offset transform to all xticklabels.
        clustermap.ax_row_dendrogram.set_visible(False)
        clustermap.ax_heatmap.tick_params(bottom=False)  # Hide xtick line

        x_labels = clustermap.ax_heatmap.xaxis.get_majorticklabels()

        dpi_x = clustermap.fig.dpi_scale_trans.to_values()[0]
        dpi_y = clustermap.fig.dpi_scale_trans.to_values()[3]

        x0 = clustermap.ax_heatmap.transData.transform(x_labels[0].get_position())
        x1 = clustermap.ax_heatmap.transData.transform(x_labels[1].get_position())

        ylims = clustermap.ax_heatmap.get_ylim()
        bottom_points = clustermap.ax_heatmap.transData.transform((1.0, ylims[0]))[1]
        for i, xl in enumerate(x_labels):
            # Move labels in dx and dy points.

            swap_xy = (1.0, xl.get_position()[0] + 0.5)
            new_y_points = clustermap.ax_heatmap.transData.transform(swap_xy)[1]

            dx = -0.5 * abs(x1[0] - x0[0]) / dpi_x
            dy = (new_y_points - bottom_points) / dpi_y

            offset = mpl.transforms.ScaledTranslation(dx, dy, clustermap.fig.dpi_scale_trans)
            xl.set_transform(xl.get_transform() + offset)

    if symmetric:
        rot = 45
    else:
        rot = 90

    va = 'center'
    clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.xaxis.get_majorticklabels(),
                                          rotation=rot,
                                          rotation_mode='anchor',
                                          va='bottom',
                                          ha='right')  # , fontsize=9.5)
    clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.yaxis.get_majorticklabels(),
                                          rotation=0,
                                          va=va,
                                          ha='left')  # , fontsize=9.5)
    return clustermap