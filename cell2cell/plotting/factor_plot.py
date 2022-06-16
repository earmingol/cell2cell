# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator
from scipy.stats import zscore

from cell2cell.clustering.cluster_interactions import compute_distance, compute_linkage
from cell2cell.analysis.tensor_downstream import get_factor_specific_ccc_networks


def context_boxplot(context_loadings, metadict, included_factors=None, group_order=None, statistical_test='Mann-Whitney',
                    pval_correction='benjamini-hochberg', text_format='star', nrows=1, figsize=(12, 6), cmap='tab10',
                    title_size=14, axis_label_size=12, group_label_rotation=45, ylabel='Context Loadings',
                    dot_color='lightsalmon', dot_edge_color='brown', filename=None, verbose=False):
    '''
    Plots a boxplot to compare the loadings of context groups in each
    of the factors resulting from a tensor decomposition.

    Parameters
    ----------
    context_loadings : pandas.DataFrame
        Dataframe containing the loadings of each of the contexts
        from a tensor decomposition. Rows are contexts and columns
        are the factors obtained.

    metadict : dict
        A dictionary containing the groups where each of the contexts
        belong to. Keys corresponds to the indexes in `context_loadings`
        and values are the respective groups. For example:
        metadict={'Context 1' : 'Group 1', 'Context 2' : 'Group 1',
                  'Context 3' : 'Group 2', 'Context 4' : 'Group 2'}

    included_factors : list, default=None
        Factors to be included. Factor names must be the same as column elements
        in the context_loadings.

    group_order : list, default=None
        Order of the groups to plot the boxplots. Considering the
        example of the metadict, it could be:
        group_order=['Group 1', 'Group 2'] or
        group_order=['Group 2', 'Group 1']
        If None, the order that groups are found in `metadict`
        will be considered.

    statistical_test : str, default='Mann-Whitney'
        The statistical test to compare context groups within each factor.
        Options include:
        't-test_ind', 't-test_welch', 't-test_paired', 'Mann-Whitney',
        'Mann-Whitney-gt', 'Mann-Whitney-ls', 'Levene', 'Wilcoxon', 'Kruskal'.

    pval_correction : str, default='benjamini-hochberg'
        Multiple test correction method to reduce false positives.
        Options include:
        'bonferroni', 'bonf', 'Bonferroni', 'holm-bonferroni', 'HB',
        'Holm-Bonferroni', 'holm', 'benjamini-hochberg', 'BH', 'fdr_bh',
        'Benjamini-Hochberg', 'fdr_by', 'Benjamini-Yekutieli', 'BY', None

    text_format : str, default='star'
        Format to display the results of the statistical test.
        Options are:

        - 'star', to display P- values < 1e-4 as "****"; < 1e-3 as "***";
                  < 1e-2 as "**"; < 0.05 as "*", and < 1 as "ns".
        - 'simple', to display P-values < 1e-5 as "1e-5"; < 1e-4 as "1e-4";
                  < 1e-3 as "0.001"; < 1e-2 as "0.01"; and < 5e-2 as "0.05".

    nrows : int, default=1
        Number of rows to generate the subplots.

    figsize : tuple, default=(12, 6)
        Size of the figure (width*height), each in inches.

    cmap : str, default='tab10'
        Name of the color palette for coloring the major groups of contexts.

    title_size : int, default=14
        Font size of the title in each of the factor boxplots.

    axis_label_size : int, default=12
        Font size of the labels for X and Y axes.

    group_label_rotation : int, default=45
        Angle of rotation for the tick labels in the X axis.

    ylabel : str, default='Context Loadings'
        Label for the Y axis.

    dot_color : str, default='lightsalmon'
        A matplotlib color for the dots representing individual contexts
        in the boxplot. For more info see:
        https://matplotlib.org/stable/gallery/color/named_colors.html

    dot_edge_color : str, default='brown'
        A matplotlib color for the edge of the dots in the boxplot.
        For more info see:
        https://matplotlib.org/stable/gallery/color/named_colors.html

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    verbose : boolean, default=None
        Whether printing out the result of the pairwise statistical tests
        in each of the factors

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure.

    axes : matplotlib.axes.Axes or array of Axes
           Matplotlib axes representing the subplots containing the boxplots.
    '''
    if group_order is not None:
        assert len(set(group_order) & set(metadict.values())) == len(set(metadict.values())), "All groups in `metadict` must be contained in `group_order`"
    else:
        group_order = list(set(metadict.values()))
    df = context_loadings.copy()

    if included_factors is None:
        factor_labels = list(df.columns)
    else:
        factor_labels = included_factors
    rank = len(factor_labels)
    df['Group'] = [metadict[idx] for idx in df.index]

    nrows = min([rank, nrows])
    ncols = int(np.ceil(rank/nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey='none')

    if rank == 1:
        axs = np.array([axes])
    else:
        axs = axes.flatten()


    for i, factor in enumerate(factor_labels):
        ax = axs[i]
        x, y = 'Group', factor

        order = group_order

        # Plot the boxes
        ax = sns.boxplot(x=x,
                         y=y,
                         data=df,
                         order=order,
                         whis=[0, 100],
                         width=.6,
                         palette=cmap,
                         boxprops=dict(alpha=.5),
                         ax=ax
                         )

        # Plot the dots
        sns.stripplot(x=x,
                      y=y,
                      data=df,
                      size=6,
                      order=order,
                      color=dot_color,
                      edgecolor=dot_edge_color,
                      linewidth=0.6,
                      jitter=False,
                      ax=ax
                      )

        if statistical_test is not None:
            # Add annotations about statistical test
            from itertools import combinations

            pairs = list(combinations(order, 2))
            annotator = Annotator(ax=ax,
                                  pairs=pairs,
                                  data=df,
                                  x=x,
                                  y=y,
                                  order=order)
            annotator.configure(test=statistical_test,
                                text_format=text_format,
                                loc='inside',
                                comparisons_correction=pval_correction,
                                verbose=verbose
                                )
            annotator.apply_and_annotate()

        ax.set_title(factor, fontsize=title_size)

        ax.set_xlabel('', fontsize=axis_label_size)
        if (i == 0) | (((i) % ncols) == 0):
            ax.set_ylabel(ylabel, fontsize=axis_label_size)
        else:
            ax.set_ylabel(' ', fontsize=axis_label_size)

        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=group_label_rotation,
                           rotation_mode='anchor',
                           va='bottom',
                           ha='right')

    # Remove extra subplots
    for j in range(i+1, axs.shape[0]):
        ax = axs[j]
        ax.axis(False)

    if axes.shape[0] > 1:
        axes = axes.reshape(axes.shape[0], -1)
        fig.align_ylabels(axes[:, 0])

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig, axes


def loading_clustermap(loadings, loading_threshold=0., use_zscore=True, metric='euclidean', method='ward',
                       optimal_leaf=True, figsize=(15, 8), heatmap_lw=0.2, cbar_fontsize=12, tick_fontsize=10, cmap=None,
                       cbar_label=None, filename=None, **kwargs):
    '''
    Plots a clustermap of the tensor-factorization loadings from one tensor dimension or
    the joint loadings from multiple tensor dimensions.

    Parameters
    ----------
    loadings : pandas.DataFrame
        Loadings for a given tensor dimension after running the tensor
        decomposition. Rows are the elements in one dimension or joint
        pairs/n-tuples in multiple dimensions. It is recommended that
        the loadings resulting from the decomposition should be
        l2-normalized prior to their use, by considering all dimensions
        together. For example, take the factors dictionary found in any
        InteractionTensor or any BaseTensor derived class, and execute
        cell2cell.tensor.normalize(factors).

    loading_threshold : float
        Threshold to filter out elements in the loadings dataframe.
        This plot considers elements with loadings greater than this
        threshold in at least one of the factors.

    use_zscore : boolean
        Whether converting loadings to z-scores across factors.

    metric : str, default='euclidean'
        The distance metric to use. The distance function can be 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.

    method : str, 'ward' by default
        Method to compute the linkage.
        It could be:

        - 'single'
        - 'complete'
        - 'average'
        - 'weighted'
        - 'centroid'
        - 'median'
        - 'ward'
        For more details, go to:
        https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.cluster.hierarchy.linkage.html

    optimal_leaf : boolean, default=True
        Whether sorting the leaf of the dendrograms to have a minimal distance
        between successive leaves. For more information, see
        scipy.cluster.hierarchy.optimal_leaf_ordering

    figsize : tuple, default=(16, 9)
        Size of the figure (width*height), each in inches.

    heatmap_lw : float, default=0.2
        Width of the lines that will divide each cell.

    cbar_fontsize : int, default=12
        Font size for the colorbar title.

    tick_fontsize : int, default=10
        Font size for ticks in the x and y axes.

    cmap : str, default=None
        Name of the color palette for coloring the heatmap. If None,
        cmap='Blues' would be used when use_zscore=False; and cmap='vlag' when use_zscore=True.

    cbar_label : str, default=None
        Label for the color bar. If None, default labels will be 'Z-scores \n across factors'
        or 'Loadings', depending on `use_zcore` is True or False, respectively.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    **kwargs : dict
        Dictionary containing arguments for the seaborn.clustermap function.

    Returns
    -------
    cm : seaborn.matrix.ClusterGrid
        A seaborn ClusterGrid instance.
    '''
    df = loadings.copy()
    df = df[(df.T > loading_threshold).any()].T
    if use_zscore:
        df = df.apply(zscore)
        if cmap is None:
            cmap = 'vlag'
        val = np.ceil(max([abs(df.min().min()), abs(df.max().max())]))
        vmin, vmax = -1. * val, val
        if cbar_label is None:
            cbar_label = 'Z-scores \n across factors'
    else:
        if cmap is None:
            cmap='Blues'
        vmin, vmax = 0., df.max().max()
        if cbar_label is None:
            cbar_label = 'Loadings'
    # Clustering
    dm_rows = compute_distance(df, axis=0, metric=metric)
    row_linkage = compute_linkage(dm_rows, method=method, optimal_ordering=optimal_leaf)

    dm_cols = compute_distance(df, axis=1, metric=metric)
    col_linkage = compute_linkage(dm_cols, method=method, optimal_ordering=optimal_leaf)

    # Clustermap
    cm = sns.clustermap(df,
                        cmap=cmap,
                        col_linkage=col_linkage,
                        row_linkage=row_linkage,
                        vmin=vmin,
                        vmax=vmax,
                        xticklabels=1,
                        figsize=figsize,
                        linewidths=heatmap_lw,
                        **kwargs
                        )

    # Color bar label
    cbar = cm.ax_heatmap.collections[0].colorbar
    cbar.ax.set_ylabel(cbar_label, fontsize=cbar_fontsize)
    cbar.ax.yaxis.set_label_position("left")

    # Tick labels
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, ha='left', fontsize=tick_fontsize)
    plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), fontsize=tick_fontsize)

    # Resize clustermap and dendrograms
    hm = cm.ax_heatmap.get_position()
    w_mult = 1.0
    h_mult = 1.0
    cm.ax_heatmap.set_position([hm.x0, hm.y0, hm.width * w_mult, hm.height])
    row = cm.ax_row_dendrogram.get_position()
    row_d_mult = 0.33
    cm.ax_row_dendrogram.set_position(
        [row.x0 + row.width * (1 - row_d_mult), row.y0, row.width * row_d_mult, row.height * h_mult])

    col = cm.ax_col_dendrogram.get_position()
    cm.ax_col_dendrogram.set_position([col.x0, col.y0, col.width * w_mult, col.height * 0.5])

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return cm


def ccc_networks_plot(factors, included_factors=None, sender_label='Sender Cells', receiver_label='Receiver Cells',
                      ccc_threshold=None, panel_size=(8, 8), nrows=2, network_layout='spring', edge_color='magenta',
                      edge_width=25, edge_arrow_size=20, edge_alpha=0.25, node_color="#210070", node_size=1000,
                      node_alpha=0.9, node_label_size=20, node_label_alpha=0.7, node_label_offset=(0.1, -0.2),
                      factor_title_size=36, filename=None):
    '''Plots factor-specific cell-cell communication networks
    resulting from decomposition with Tensor-cell2cell.

    Parameters
    ----------
    factors : dict
        Ordered dictionary containing a dataframe with the factor loadings for each
        dimension/order of the tensor.

    included_factors : list, default=None
        Factors to be included. Factor names must be the same as the key values in
        the factors dictionary.

    sender_label : str
        Label for the dimension of sender cells. It is one key of the factors dict.

    receiver_label : str
        Label for the dimension of receiver cells. It is one key of the factors dict.

    ccc_threshold : float, default=None
        Threshold to consider only edges with a higher weight than this value.

    panel_size : tuple, default=(8, 8)
        Size of one subplot or network (width*height), each in inches.

    nrows : int, default=2
        Number of rows in the set of subplots.

    network_layout : str, default='spring'
        Visualization layout of the networks. It uses algorithms implemented
        in NetworkX, including:
            -'spring' : Fruchterman-Reingold force-directed algorithm.
            -'circular' : Position nodes on a circle.

    edge_color : str, default='magenta'
        Color of the edges in the network.

    edge_width : int, default=25
        Thickness of the edges in the network.

    edge_arrow_size : int, default=20
        Size of the arrow of an edge pointing towards the receiver cells.

    edge_alpha : float, default=0.25
        Transparency of the edges. Values must be between 0 and 1. Higher
        values indicates less transparency.

    node_color : str, default="#210070"
        Color of the nodes in the network.

    node_size : int, default=1000
        Size of the nodes in the network.

    node_alpha : float, default=0.9
        Transparency of the nodes. Values must be between 0 and 1. Higher
        values indicates less transparency.

    node_label_size : int, default=20
        Size of the labels for the node names.

    node_label_alpha : int, default=0.7
        Transparency of the node labeks. Values must be between 0 and 1.
        Higher values indicates less transparency.

    node_label_offset : tuple, default=(0.1, -0.2)
        Offset values to move the node labels away from the center of the nodes.

    factor_title_size : int, default=36
        Size of the subplot titles. Each network has a title like 'Factor 1',
        'Factor 2', ... ,'Factor R'.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure.

    axes : matplotlib.axes.Axes or array of Axes
        Matplotlib axes representing the subplots containing the networks.
    '''
    networks = get_factor_specific_ccc_networks(result=factors,
                                                sender_label=sender_label,
                                                receiver_label=receiver_label)

    if included_factors is None:
        factor_labels = [f'Factor {i}' for i in range(1, len(networks) + 1)]
    else:
        factor_labels = included_factors

    nrows = min([len(factor_labels), nrows])
    ncols = int(np.ceil(len(factor_labels) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    if len(factor_labels) == 1:
        axs = np.array([axes])
    else:
        axs = axes.flatten()

    for i, factor in enumerate(factor_labels):
        ax = axs[i]
        if ccc_threshold is not None:
            # Considers edges with weight above ccc_threshold
            df = networks[factor].gt(ccc_threshold).astype(int).multiply(networks[factor])
        else:
            df = networks[factor]

        # Networkx Directed Network
        G = nx.convert_matrix.from_pandas_adjacency(df, create_using=nx.DiGraph())

        # Layout for visualization - Node positions
        if network_layout == 'spring':
            pos = nx.spring_layout(G,
                                   k=1.,
                                   seed=888
                                   )
        elif network_layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            raise ValueError("network_layout should be either 'spring' or 'circular'")

        # Weights for edge thickness
        weights = np.asarray([G.edges[e]['weight'] for e in G.edges()])

        # Visualize network
        nx.draw_networkx_edges(G,
                               pos,
                               alpha=edge_alpha,
                               arrowsize=edge_arrow_size,
                               width=weights * edge_width,
                               edge_color=edge_color,
                               connectionstyle="arc3,rad=-0.3",
                               ax=ax
                               )
        nx.draw_networkx_nodes(G,
                               pos,
                               node_color=node_color,
                               node_size=node_size,
                               alpha=node_alpha,
                               ax=ax
                               )
        label_options = {"ec": "k", "fc": "white", "alpha": node_label_alpha}
        _ = nx.draw_networkx_labels(G,
                                    {k: v + np.array(node_label_offset) for k, v in pos.items()},
                                    font_size=node_label_size,
                                    bbox=label_options,
                                    ax=ax
                                    )

        ax.set_frame_on(False)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        coeff = 1.4
        ax.set_xlim((xlim[0] * coeff, xlim[1] * coeff))
        ax.set_ylim((ylim[0] * coeff, ylim[1] * coeff))
        ax.set_title(factor, fontsize=factor_title_size, fontweight='bold')

    # Remove extra subplots
    for j in range(i+1, axs.shape[0]):
        ax = axs[j]
        ax.axis(False)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig, axes