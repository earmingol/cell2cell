# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from cell2cell.plotting.aesthetics import generate_legend, get_colors_from_labels, map_colors_to_metadata
from cell2cell.preprocessing.signal import smooth_curve


def tensor_factors_plot(interaction_tensor, order_labels=None, reorder_elements=None, metadata=None,
                        sample_col='Element', group_col='Category', meta_cmaps=None, fontsize=20, plot_legend=True,
                        filename=None):
    '''Plots the loadings for each element in each dimension of the tensor, generate by
    a tensor factorization.

    Parameters
    ----------
    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor generated with any of the tensor class in
        cell2cell.tensor.

    order_labels : list, default=None
        List with the labels of each dimension to use in the plot. If none, the
        default names given when factorizing the tensor will be used.

    reorder_elements : dict, default=None
        Dictionary for reordering elements in each of the tensor dimension.
        Keys of this dictionary could be any or all of the keys in
        interaction_tensor.factors. Values are list with the names or labels of the
        elements in a tensor dimension. For example, for the context dimension,
        all elements included in interaction_tensor.factors['Context'].index must
        be present.

    metadata : list, default=None
        List of pandas dataframes with metadata information for elements of each
        dimension in the tensor. A column called as the variable `sample_col` contains
        the name of each element in the tensor while another column called as the
        variable `group_col` contains the metadata or grouping information of each
        element.

    sample_col : str, default='Element'
        Name of the column containing the element names in the metadata.

    group_col : str, default='Category'
        Name of the column containing the metadata or grouping information for each
        element in the metadata.

    meta_cmaps : list, default=None
        A list of colormaps used for coloring elements in each dimension. The length
        of this list is equal to the number of dimensions of the tensor. If None, all
        dimensions will be colores with the colormap 'gist_rainbow'.

    fontsize : int, default=20
        Font size of the tick labels. Axis labels will be 1.2 times the fontsize.

    plot_legend : boolean, default=True
        Whether plotting the legends for the coloring of each element in their
        respective dimensions.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is
        not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object made with matplotlib

    axes : matplotlib.axes.Axes or array of Axes
        List of Axes for each subplot in the figure.
    '''
    # Prepare inputs for matplotlib
    assert interaction_tensor.factors is not None, "First run the method 'compute_tensor_factorization' in your InteractionTensor"
    dim = len(interaction_tensor.factors)

    if order_labels is not None:
        assert dim == len(order_labels), "The lenght of factor_labels must match the order of the tensor (order {})".format(dim)
    else:
        order_labels = list(interaction_tensor.factors.keys())

    rank = interaction_tensor.rank
    fig, axes = tensor_factors_plot_from_loadings(factors=interaction_tensor.factors,
                                                  rank=rank,
                                                  order_labels=order_labels,
                                                  reorder_elements=reorder_elements,
                                                  metadata=metadata,
                                                  sample_col=sample_col,
                                                  group_col=group_col,
                                                  meta_cmaps=meta_cmaps,
                                                  fontsize=fontsize,
                                                  plot_legend=plot_legend,
                                                  filename=filename)
    return fig, axes


def tensor_factors_plot_from_loadings(factors, rank=None, order_labels=None, reorder_elements=None, metadata=None,
                                      sample_col='Element', group_col='Category', meta_cmaps=None, fontsize=20, plot_legend=True,
                                      filename=None):
    '''Plots the loadings for each element in each dimension of the tensor, generate by
    a tensor factorization.

    Parameters
    ----------
    factors : collections.OrderedDict
        An ordered dictionary wherein keys are the names of each
        tensor dimension, and values are the loadings in a pandas.DataFrame.
        In this dataframe, rows are the elements of the respective dimension
        and columns are the factors from the tensor factorization. Values
        are the corresponding loadings.

    rank : int, default=None
        Number of factors generated from the decomposition

    order_labels : list, default=None
        List with the labels of each dimension to use in the plot. If none, the
        default names given when factorizing the tensor will be used.

    reorder_elements : dict, default=None
        Dictionary for reordering elements in each of the tensor dimension.
        Keys of this dictionary could be any or all of the keys in
        interaction_tensor.factors. Values are list with the names or labels of the
        elements in a tensor dimension. For example, for the context dimension,
        all elements included in interaction_tensor.factors['Context'].index must
        be present.

    metadata : list, default=None
        List of pandas dataframes with metadata information for elements of each
        dimension in the tensor. A column called as the variable `sample_col` contains
        the name of each element in the tensor while another column called as the
        variable `group_col` contains the metadata or grouping information of each
        element.

    sample_col : str, default='Element'
        Name of the column containing the element names in the metadata.

    group_col : str, default='Category'
        Name of the column containing the metadata or grouping information for each
        element in the metadata.

    meta_cmaps : list, default=None
        A list of colormaps used for coloring elements in each dimension. The length
        of this list is equal to the number of dimensions of the tensor. If None, all
        dimensions will be colores with the colormap 'gist_rainbow'.

    fontsize : int, default=20
        Font size of the tick labels. Axis labels will be 1.2 times the fontsize.

    plot_legend : boolean, default=True
        Whether plotting the legends for the coloring of each element in their
        respective dimensions.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is
        not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object made with matplotlib

    axes : matplotlib.axes.Axes or array of Axes
        List of Axes for each subplot in the figure.
    '''
    # Prepare inputs for matplotlib
    if rank is not None:
        assert list(factors.values())[0].shape[1] == rank, "Rank must match the number of columns in dataframes in `factors`"
    else:
        rank = list(factors.values())[0].shape[1]

    dim = len(factors)

    if order_labels is not None:
        assert dim == len(order_labels), "The length of factor_labels must match the order of the tensor (order {})".format(dim)
    else:
        order_labels = list(factors.keys())

    if metadata is not None:
        meta_og = metadata.copy()
    if reorder_elements is not None:
        factors, metadata = reorder_dimension_elements(factors=factors,
                                                       reorder_elements=reorder_elements,
                                                       metadata=metadata)

    if metadata is None:
        metadata = [None] * dim
        meta_colors = [None] * dim
        element_colors = [None] * dim
    else:
        if meta_cmaps is None:
            meta_cmaps = ['gist_rainbow']*len(metadata)
        assert len(metadata) == len(meta_cmaps), "Provide a cmap for each order"
        assert len(metadata) == len(factors), "Provide a metadata for each order. If there is no metadata for any, replace with None"
        meta_colors = [get_colors_from_labels(m[group_col], cmap=cmap) if ((m is not None) & (cmap is not None)) else None for m, cmap in zip(meta_og, meta_cmaps)]
        element_colors = [map_colors_to_metadata(metadata=m,
                                                 colors=mc,
                                                 sample_col=sample_col,
                                                 group_col=group_col,
                                                 cmap=cmap).to_dict() if ((m is not None) & (cmap is not None)) else None for m, cmap, mc in zip(metadata, meta_cmaps, meta_colors)]

    # Make the plot
    fig, axes = plt.subplots(nrows=rank,
                             ncols=dim,
                             figsize=(10, int(rank * 1.2 + 1)),
                             sharex='col',
                             #sharey='col'
                             )

    axes = axes.reshape((rank, dim))

    # Factor by factor
    if rank > 1:
        # Iterates horizontally (dimension by dimension)
        for ind, (order_factors, axs) in enumerate(zip(factors.values(), axes.T)):
            if isinstance(order_factors, pd.Series):
                order_factors = order_factors.to_frame().T
            # Iterates vertically (factor by factor)
            for i, (df_row, ax) in enumerate(zip(order_factors.T.iterrows(), axs)):
                factor_name = df_row[0]
                factor = df_row[1]
                sns.despine(top=True, ax=ax)
                if (metadata[ind] is not None) & (meta_colors[ind] is not None):
                    plot_colors = [element_colors[ind][idx] for idx in order_factors.index]
                    ax.bar(range(len(factor)), factor.values.tolist(), color=plot_colors)
                else:
                    ax.bar(range(len(factor)), factor.values.tolist())
                axes[i, 0].set_ylabel(factor_name, fontsize=int(1.2*fontsize))
                if i < len(axs):
                    ax.tick_params(axis='x', which='both', length=0)
                    ax.tick_params(axis='both', labelsize=fontsize)
                    plt.setp(ax.get_xticklabels(), visible=False)
            axs[-1].set_xlabel(order_labels[ind], fontsize=int(1.2*fontsize), labelpad=fontsize)
    else:
        for ind, order_factors in enumerate(factors.values()):
            if isinstance(order_factors, pd.Series):
                order_factors = order_factors.to_frame().T
            ax = axes[ind]
            ax.set_xlabel(order_labels[ind], fontsize=int(1.2*fontsize), labelpad=fontsize)
            for i, df_row in enumerate(order_factors.T.iterrows()):
                factor_name = df_row[0]
                factor = df_row[1]
                sns.despine(top=True, ax=ax)
                if (metadata[ind] is not None) & (meta_colors[ind] is not None):
                    plot_colors = [element_colors[ind][idx] for idx in order_factors.index]
                    ax.bar(range(len(factor)), factor.values.tolist(), color=plot_colors)
                else:
                    ax.bar(range(len(factor)), factor.values.tolist())
                ax.set_ylabel(factor_name, fontsize=int(1.2*fontsize))

    fig.align_ylabels(axes[:,0])
    plt.tight_layout()

    # Include legends of coloring the elements in each dimension.
    if plot_legend:
        # Set current axis:
        plt.sca(axes[0, -1])

        # Legends
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        count = 0
        for ind, order in enumerate(order_labels):
            if (metadata[ind] is not None) & (meta_colors[ind] is not None):
                if count == 0:
                    bbox_cords =  (1.05, 1.2)
                lgd = generate_legend(color_dict=meta_colors[ind],
                                      bbox_to_anchor=bbox_cords,
                                      loc='upper left',
                                      title=order_labels[ind],
                                      fontsize=fontsize,
                                      sorted_labels=False,
                                      )
                cords = lgd.get_window_extent(renderer).transformed(axes[0, -1].transAxes.inverted())
                xrange = abs(cords.p0[0] - cords.p1[0])
                bbox_cords = (bbox_cords[0] + xrange + 0.05, bbox_cords[1])
                count += 1

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return fig, axes


def reorder_dimension_elements(factors, reorder_elements, metadata=None):
    '''Reorders elements in the dataframes including factor loadings.

    Parameters
    ----------
    factors : dict
        Ordered dictionary containing a dataframe with the factor loadings for each
        dimension/order of the tensor.

    reorder_elements : dict, default=None
        Dictionary for reordering elements in each of the tensor dimension.
        Keys of this dictionary could be any or all of the keys in
        interaction_tensor.factors. Values are list with the names or labels of the
        elements in a tensor dimension. For example, for the context dimension,
        all elements included in interaction_tensor.factors['Context'].index must
        be present.

    metadata : list, default=None
        List of pandas dataframes with metadata information for elements of each
        dimension in the tensor. A column called as the variable `sample_col` contains
        the name of each element in the tensor while another column called as the
        variable `group_col` contains the metadata or grouping information of each
        element.

    Returns
    -------
    reordered_factors : dict
        Ordered dictionary containing a dataframe with the factor loadings for each
        dimension/order of the tensor. This dictionary includes the new orders.

    new_metadata : list, default=None
        List of pandas dataframes with metadata information for elements of each
        dimension in the tensor. A column called as the variable `sample_col` contains
        the name of each element in the tensor while another column called as the
        variable `group_col` contains the metadata or grouping information of each
        element. In this case, elements are sorted according to reorder_elements.

    '''
    assert all(k in factors.keys() for k in reorder_elements.keys()), "Keys in 'reorder_elements' must be only keys in 'factors'"
    assert all((len(set(factors[key].index).difference(set(reorder_elements[key]))) == 0) for key in reorder_elements.keys()), "All elements of each dimension included should be present"

    reordered_factors = factors.copy()
    new_metadata = metadata.copy()

    i = 0
    for k, df in reordered_factors.items():
        if k in reorder_elements.keys():
            df = df.loc[reorder_elements[k]]
            reordered_factors[k] = df[~df.index.duplicated(keep='first')]
            if new_metadata is not None:
                meta = new_metadata[i]
                meta['Element'] = pd.Categorical(meta['Element'], ordered=True, categories=list(reordered_factors[k].index))
                new_metadata[i] = meta.sort_values(by='Element').reset_index(drop=True)
        else:
            reordered_factors[k] = df
        i += 1
    return reordered_factors, new_metadata


def plot_elbow(loss, elbow=None, figsize=(4, 2.25), ylabel='Normalized Error', fontsize=14, filename=None):
    '''Plots the errors of an elbow analysis with just one run of a tensor factorization
    for each rank.

    Parameters
    ----------
    loss : list
        List of  tuples with (x, y) coordinates for the elbow analysis. X values are
        the different ranks and Y values are the errors of each decomposition.

    elbow : int, default=None
        X coordinate to color the error as red. Usually used to represent the detected
        elbow.

    figsize : tuple, default=(4, 2.25)
        Figure size, width by height

    ylabel : str, default='Normalized Error'
        Label for the y-axis

    fontsize : int, default=14
        Fontsize for axis labels.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object made with matplotlib
    '''

    fig = plt.figure(figsize=figsize)

    plt.plot(*zip(*loss))
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.xlabel('Rank', fontsize=int(1.2*fontsize))
    plt.ylabel(ylabel, fontsize=int(1.2 * fontsize))

    if elbow is not None:
        _ = plt.plot(*loss[elbow - 1], 'ro')

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return fig


def plot_multiple_run_elbow(all_loss, elbow=None, ci='95%', figsize=(4, 2.25), ylabel='Normalized Error', fontsize=14,
                            smooth=False, filename=None):
    '''Plots the errors of an elbow analysis with multiple runs of a tensor
    factorization for each rank.

    Parameters
    ----------
    all_loss : ndarray
        Array containing the errors associated with multiple runs for a given rank.
        This array is of shape (runs, upper_rank).

    elbow : int, default=None
        X coordinate to color the error as red. Usually used to represent the detected
        elbow.

    ci : str, default='std'
        Confidence interval for representing the multiple runs in each rank.
        {'std', '95%'}

    figsize : tuple, default=(4, 2.25)
        Figure size, width by height

    ylabel : str, default='Normalized Error'
        Label for the y-axis

    fontsize : int, default=14
        Fontsize for axis labels.

    smooth : boolean, default=False
        Whether smoothing the curve with a Savitzky-Golay filter.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object made with matplotlib
    '''
    fig = plt.figure(figsize=figsize)

    x = list(range(1, all_loss.shape[1]+1))
    mean = np.nanmean(all_loss, axis=0)
    std = np.nanstd(all_loss, axis=0)

    if smooth:
        mean = smooth_curve(mean)

    # Plot Mean
    plt.plot(x, mean, 'ob')

    # Plot CI
    if ci == '95%':
        coeff = 1.96
    elif ci == 'std':
        coeff = 1.0
    else:
        raise ValueError("Specify a correct ci. Either '95%' or 'std'")

    plt.fill_between(x, mean-coeff*std, mean+coeff*std, color='steelblue', alpha=.2,
                     label='$\pm$ 1 std')


    plt.tick_params(axis='both', labelsize=fontsize)
    plt.xlabel('Rank', fontsize=int(1.2*fontsize))
    plt.ylabel(ylabel, fontsize=int(1.2 * fontsize))

    if elbow is not None:
        _ = plt.plot(x[elbow - 1], mean[elbow - 1], 'ro')

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return fig


def generate_plot_df(interaction_tensor):
    '''Generates a melt dataframe with loadings for each element in all dimensions
    across factors

    Parameters
    ----------
    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor generated with any of the tensor class in
        cell2cell.tensor

    Returns
    -------
    plot_df : pandas.DataFrame
        A dataframe containing loadings for every element of all dimensions across
        factors from the decomposition. Rows are loadings individual elements of each
        dimension in a given factor, while columns are the following list
        ['Factor', 'Variable', 'Value', 'Order']
    '''
    tensor_dim = len(interaction_tensor.tensor.shape)
    if interaction_tensor.order_labels is None:
        if tensor_dim == 4:
            factor_labels = ['Context', 'LRs', 'Sender', 'Receiver']
        elif tensor_dim > 4:
            factor_labels = ['Context-{}'.format(i + 1) for i in range(tensor_dim - 3)] + ['LRs', 'Sender', 'Receiver']
        elif tensor_dim == 3:
            factor_labels = ['LRs', 'Sender', 'Receiver']
        else:
            raise ValueError('Too few dimensions in the tensor')
    else:
        assert len(interaction_tensor.order_labels) == tensor_dim, "The length of order_labels must match the number of orders/dimensions in the tensor"
        factor_labels = interaction_tensor.order_labels
    plot_df = pd.DataFrame()
    for lab, order_factors in enumerate(interaction_tensor.factors.values()):
        sns_df = order_factors.T
        sns_df.index.name = 'Factors'
        melt_df = pd.melt(sns_df.reset_index(), id_vars=['Factors'], value_vars=sns_df.columns)
        melt_df = melt_df.assign(Order=factor_labels[lab])

        plot_df = pd.concat([plot_df, melt_df])
    plot_df.columns = ['Factor', 'Variable', 'Value', 'Order']

    return plot_df