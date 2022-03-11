import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator
from scipy.stats import zscore

from cell2cell.clustering.cluster_interactions import compute_distance, compute_linkage


def context_boxplot(context_loadings, metadict, group_order=None, statistical_test='Mann-Whitney',
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

    y_label : str, default='Context Loadings'
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

    ax : matplotlib.axes.Axes or array of Axes
        Matplotlib axes representing the subplots containing the boxplots.
    '''
    if group_order is not None:
        assert len(set(group_order) & set(metadict.values())) == len(set(metadict.values())), "All groups in `metadict` must be contained in `group_order`"
    else:
        group_order = list(set(metadict.values()))
    df = context_loadings.copy()
    rank = df.shape[1]
    df['Group'] = [metadict[idx] for idx in df.index]

    ncols = int(np.ceil(rank/nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey='none')
    axs = axes.flatten()
    for i in range(1, rank + 1):
        ax = axs[i - 1]

        factor = 'Factor {}'.format(i)
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
        if (i == 1) | (((i-1) % ncols) == 0):
            ax.set_ylabel(ylabel, fontsize=axis_label_size)
        else:
            ax.set_ylabel(' ', fontsize=axis_label_size)

        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=group_label_rotation,
                           rotation_mode='anchor',
                           va='bottom',
                           ha='right')

    # Remove extra subplots
    for j in range(i, axs.shape[0]):
        ax = axs[j]
        ax.axis(False)

    if axes.shape[0] > 1:
        fig.align_ylabels(axes[:, 0])

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig, axes


def loading_clustermap(loadings, loading_threshold=0., use_zscore=True, metric='euclidean', method='ward',
                       optimal_leaf=True, figsize=(15, 8), heatmap_lw=0.2, cbar_fontsize=12, tick_fontsize=10, cmap=None,
                       filename=None, **kwargs):
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
        Method to compute the linkage. It could be:
        'single'
        'complete'
        'average'
        'weighted'
        'centroid'
        'median'
        'ward'
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
        cbar_label = 'Z-scores \n across factors'
    else:
        if cmap is None:
            cmap='Blues'
        vmin, vmax = 0., df.max().max()
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