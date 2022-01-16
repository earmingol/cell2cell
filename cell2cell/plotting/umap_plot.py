# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt


def umap_biplot(umap_df, figsize=(8 ,8), ax=None, show_axes=True, show_legend=True, hue=None,
                cmap='tab10', fontsize=20, filename=None):
    '''Plots a UMAP biplot for the UMAP embeddings.

    Parameters
    ----------
    umap_df : pandas.DataFrame
        Dataframe containing the UMAP embeddings for the axis analyzed.
        It must contain columns 'umap1 and 'umap2'. If a hue column is
        provided in the parameter 'hue', that column must be provided
        in this dataframe.

    figsize : tuple, default=(8, 8)
        Size of the figure (width*height), each in inches.

    ax : matplotlib.axes.Axes, default=None
        The matplotlib axes containing a plot.

    show_axes : boolean, default=True
        Whether showing lines, ticks and ticklabels of both axes.

    show_legend : boolean, default=True
        Whether including the legend when a hue is provided.

    hue : vector or key in 'umap_df'
        Grouping variable that will produce points with different colors.
        Can be either categorical or numeric, although color mapping will
        behave differently in latter case.

    cmap : str, default='tab10'
        Name of the color palette for coloring elements with UMAP embeddings.

    fontsize : int, default=20
        Fontsize of the axis labels (UMAP1 and UMAP2).

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    Returns
    -------
        fig : matplotlib.figure.Figure
            A matplotlib Figure instance.

        ax : matplotlib.axes.Axes
            The matplotlib axes containing the plot.
    '''

    if ax is None:
        fig = plt.figure(figsize=figsize)

    ax = sns.scatterplot(x='umap1',
                         y='umap2',
                         data=umap_df,
                         hue=hue,
                         palette=cmap,
                         ax=ax
                         )

    if show_axes:
        sns.despine(ax=ax,
                    offset=15
                    )

        ax.tick_params(axis='both',
                       which='both',
                       colors='black',
                       width=2,
                       length=5
                       )
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for key, spine in ax.spines.items():
            spine.set_visible(False)


    for tick in ax.get_xticklabels():
        tick.set_fontproperties('arial')
        tick.set_weight("bold")
        tick.set_color("black")
        tick.set_fontsize(int(0.7*fontsize))
    for tick in ax.get_yticklabels():
        tick.set_fontproperties('arial')
        tick.set_weight("bold")
        tick.set_color("black")
        tick.set_fontsize(int(0.7*fontsize))

    ax.set_xlabel('UMAP 1', fontsize=fontsize)
    ax.set_ylabel('UMAP 2', fontsize=fontsize)

    if (show_legend) & (hue is not None):
        # Put the legend out of the figure
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        legend.set_title(hue)
        legend.get_title().set_fontsize(int(0.7*fontsize))

        for text in legend.get_texts():
            text.set_fontsize(int(0.7*fontsize))

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if ax is None:
        return fig, ax
    else:
        return ax