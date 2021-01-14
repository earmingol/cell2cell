# -*- coding: utf-8 -*-

import matplotlib as mlp
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from cell2cell.clustering import compute_linkage
from cell2cell.preprocessing.manipulate_dataframes import check_symmetry
from cell2cell.plotting.aesthetics import get_colors_from_labels


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
    symmetric = check_symmetry(df)
    if symmetric:
        if 'col_cluster' in kwargs.keys():
            kwargs['row_cluster'] = kwargs['col_cluster']
            if kwargs['col_cluster']:
                linkage = compute_linkage(df, method=method, optimal_ordering=optimal_leaf)
            else:
                linkage = None
        else:
            linkage = compute_linkage(df, method=method, optimal_ordering=optimal_leaf)


        # Triangular matrix
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

        kwargs_ = kwargs.copy()
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