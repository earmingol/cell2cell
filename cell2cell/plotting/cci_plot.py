# -*- coding: utf-8 -*-

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
                   cbar_title='CCI score', cbar_fontsize=18, filename=None, **kwargs):
    '''Generates a clustermap (heatmap + dendrograms from a hierarchical
    clustering) based on CCI scores of cell-cell pairs.

    Parameters
    ----------
    interaction_space : cell2cell.core.interaction_space.InteractionSpace
        Interaction space that contains all a distance matrix after running the
        the method compute_pairwise_cci_scores. Alternatively, this object
        can be a numpy-array or a pandas DataFrame. Also, a
        SingleCellInteractions or a BulkInteractions object after running
        the method compute_pairwise_cci_scores.

    method : str, default='ward'
        Clustering method for computing a linkage as in
        scipy.cluster.hierarchy.linkage

    optimal_leaf : boolean, default=True
        Whether sorting the leaf of the dendrograms to have a minimal distance
        between successive leaves. For more information, see
        scipy.cluster.hierarchy.optimal_leaf_ordering

    metadata : pandas.Dataframe, default=None
        Metadata associated with the cells, cell types or samples in the
        matrix containing CCI scores. If None, cells will not be colored
        by major groups.

    sample_col : str, default='#SampleID'
        Column in the metadata for the cells, cell types or samples
        in the matrix containing CCI scores.

    group_col : str, default='Groups'
        Column in the metadata containing the major groups of cells, cell types
        or samples in the matrix with CCI scores.

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
        Title of the clustermap.

    cbar_title : str, default='CCI score'
        Title for the colorbar, depending on the score employed.

    cbar_fontsize : int, default=18
        Font size for the colorbar title as well as labels for axes X and Y.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    **kwargs : dict
        Dictionary containing arguments for the seaborn.clustermap function.

    Returns
    -------
    hier : seaborn.matrix.ClusterGrid
        A seaborn ClusterGrid instance.
    '''
    if hasattr(interaction_space, 'distance_matrix'):
        print('Interaction space detected as an InteractionSpace class')
        distance_matrix = interaction_space.distance_matrix
        space_type = 'class'
    elif (type(interaction_space) is np.ndarray) or (type(interaction_space) is pd.core.frame.DataFrame):
        print('Interaction space detected as a distance matrix')
        distance_matrix = interaction_space
        space_type = 'matrix'
    elif hasattr(interaction_space, 'interaction_space'):
        print('Interaction space detected as a Interactions class')
        if not hasattr(interaction_space.interaction_space, 'distance_matrix'):
            raise ValueError('First run the method compute_pairwise_interactions() in your interaction' + \
                             ' object to generate a distance matrix.')
        else:
            interaction_space = interaction_space.interaction_space
            distance_matrix = interaction_space.distance_matrix
            space_type = 'class'
    else:
        raise ValueError('First run the method compute_pairwise_interactions() in your interaction' + \
                         ' object to generate a distance matrix.')

    # Drop excluded cells
    if excluded_cells is not None:
        df = distance_matrix.loc[~distance_matrix.index.isin(excluded_cells),
                                 ~distance_matrix.columns.isin(excluded_cells)].copy()
    else:
        df = distance_matrix.copy()

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
        col_colors = map_colors_to_metadata(metadata=metadata,
                                            ref_df=df,
                                            colors=colors,
                                            sample_col=sample_col,
                                            group_col=group_col,
                                            cmap=meta_cmap)

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

    if ~symmetric:
        hier.ax_heatmap.set_xlabel('Receiver cells', fontsize=cbar_fontsize)
        hier.ax_heatmap.set_ylabel('Sender cells', fontsize=cbar_fontsize)

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return hier


def _get_distance_matrix_linkages(df, kwargs, method='ward', optimal_ordering=True, symmetric=None):
    '''Computes linkages for the CCI matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains the CCI scores in a form of distances (that is, smaller
        values represent stronger interactions). Diagonal must be filled
        by zeros.

    kwargs : dict
        Dictionary containing arguments for the seaborn.clustermap function.

    method : str, default='ward'
        Clustering method for computing a linkage as in
        scipy.cluster.hierarchy.linkage

    optimal_ordering : boolean, default=True
        Whether sorting the leaf of the dendrograms to have a minimal distance
        between successive leaves. For more information, see
        scipy.cluster.hierarchy.optimal_leaf_ordering

    symmetric : boolean, default=None
        Whether df is symmetric.

    Returns
    -------
    linkage : ndarray
        The hierarchical clustering of cells encoded as a linkage matrix.
    '''
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
    '''Generates a mask to plot the upper triangle of the CCI matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains the CCI scores. Must be a symmetric matrix.

    linkage : ndarray, default=None
        The hierarchical clustering of cells encoded as a linkage matrix.

    symmetric : boolean, default=None
        Whether df is symmetric.

    **kwargs : dict
        Dictionary containing arguments for the seaborn.clustermap function.

    Returns
    -------
    mask : ndarray
        Mask that contains ones in the places to be hidden in the clustermap.
        Only the diagonal and the upper triangle are not masked (contain
        zeros).
    '''
    if symmetric is None:
        symmetric = check_symmetry(df)

    # Triangular matrix
    if symmetric:

        if linkage is None:
            hier = sns.clustermap(df,
                                  **kwargs
                                  )
        else:
            # Plot hierarchical clustering for getting indexes according to linkage
            hier = sns.clustermap(df,
                                  col_linkage=linkage,
                                  row_linkage=linkage,
                                  **kwargs
                                  )
        plt.close()

        # Get indexes for elements given dendrogram order
        if 'col_cluster' in kwargs.keys():
            if kwargs['col_cluster']:
                col_order = hier.dendrogram_col.reordered_ind
            else:
                col_order = range(df.shape[1])
        else:
            col_order = hier.dendrogram_col.reordered_ind

        if 'row_cluster' in kwargs.keys():
            if kwargs['row_cluster']:
                row_order = hier.dendrogram_row.reordered_ind
            else:
                row_order = range(df.shape[0])
        else:
            row_order = hier.dendrogram_row.reordered_ind

        # Generate mask
        mask = np.zeros((df.shape[0], df.shape[1]))
        for i, ind in enumerate(col_order):
            filter_list = [row_order[j] for j in range(i)]
            mask[ind, filter_list] = 1
    else:
        mask = None
    return mask


def _plot_triangular_clustermap(df, symmetric=None, linkage=None, mask=None, col_colors=None, row_colors=None,
                                title='', cbar_title='CCI score', cbar_fontsize=12, **kwargs):
    '''Plots a triangular clustermap based on a mask.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains the CCI scores. Must be a symmetric matrix.

    linkage : ndarray, default=None
        The hierarchical clustering of cells encoded as a linkage matrix.

    mask : ndarray, default=None
        Mask that contains ones in the places to be hidden in the clustermap.
        Only the diagonal and the upper triangle are not masked (contain
        zeros). If None, a mask will be computed based on the CCI matrix
        symmetry.

    col_colors : dict, default=None
        Dictionary containing tuples in the RGBA format for indicating colors
        of major groups of cells in the columns.

    row_colors : dict, default=None
        Dictionary containing tuples in the RGBA format for indicating colors
        of major groups of cells in the rows.

    title : str, default=''
        Title of the clustermap.

    cbar_title : str, default='CCI score'
        Title for the colorbar, depending on the score employed.

    cbar_fontsize : int, default=12
        Font size for the colorbar title as well as labels for axes X and Y.

    **kwargs : dict
        Dictionary containing arguments for the seaborn.clustermap function.

    Returns
    -------
    hier : seaborn.matrix.ClusterGrid
        A seaborn ClusterGrid instance.
    '''
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
    '''Moves xticks to the diagonal when plotting a symmetric matrix
    in the form of a upper triangle.

    Parameters
    ---------
    clustermap : seaborn.matrix.ClusterGrid
        A seaborn ClusterGrid instance.

    symmetric : boolean, default=None
        Whether the CCI matrix plotted in the clustermap is symmetric.

    Returns
    -------
    clustermap : seaborn.matrix.ClusterGrid
        A seaborn ClusterGrid instance, with the xticks moved to the
        diagonal if the CCI matrix was symmetric. If not, the same
        input clustermap is returned, but with rotated xtick labels.
    '''
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