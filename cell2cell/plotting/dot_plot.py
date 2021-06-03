# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def dot_plot(sc_interactions, evaluation='communication', significance=0.05, senders=None, receivers=None,
             figsize=(16, 9), tick_size=8, cmap='PuOr', filename=None):
    '''Generates a dot plot for the CCI or communication scores given their
    P-values. Size of the dots are given by the -log10(P-value) and colors
    by the value of the CCI or communication score.

    Parameters
    ----------
    sc_interactions : cell2cell.analysis.pipelines.SingleCellInteractions
        Interaction class with all necessary methods to run the cell2cell
        pipeline on a single-cell RNA-seq dataset. The method
        permute_cell_labels() must be run before generating this plot.

    evaluation : str, default='communication'
        P-values of CCI or communication scores used for this plot.
        - 'interactions' : For CCI scores
        - 'communication' : For communication scores

    significance : float, default=0.05
        The significance threshold to be plotted. LR pairs or cell-cell
        pairs with at least one P-value below this threshold will be
        considered.

    senders : list, default=None
        Optional filter to plot specific sender cells.

    receivers : list, default=None
        Optional filter to plot specific receiver cells.

    figsize : tuple, default=(16, 9)
        Size of the figure (width*height), each in inches.

    tick_size : int, default=8
        Specifies the size of ticklabels as well as the maximum size
        of the dots.

    cmap : str, default='PuOr'
        A matplotlib color palette name.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object made with matplotlib
    '''
    if evaluation == 'communication':
        if not (hasattr(sc_interactions, 'ccc_permutation_pvalues')):
            raise ValueError(
                'Run the method permute_cell_labels() with evaluation="communication" before plotting communication P-values.')
        else:
            pvals = sc_interactions.ccc_permutation_pvalues
            scores = sc_interactions.interaction_space.interaction_elements['communication_matrix']
    elif evaluation == 'interactions':
        if not (hasattr(sc_interactions, 'cci_permutation_pvalues')):
            raise ValueError(
                'Run the method permute_cell_labels() with evaluation="interactions" before plotting interaction P-values.')
        else:
            pvals = sc_interactions.cci_permutation_pvalues
            scores = sc_interactions.interaction_space.interaction_elements['cci_matrix']
    else:
        raise ValueError('evaluation has to be either "communication" or "interactions"')

    # Preprocessing
    norm = mpl.colors.Normalize(vmin=np.min(np.min(scores)), vmax=np.max(np.max(scores)))
    df = pvals.lt(significance).astype(int)

    # Filter cells
    if evaluation == 'communication':
        if (senders is not None) and (receivers is not None):
            new_cols = [s + ';' + r for r in receivers for s in senders]
        elif senders is not None:
            new_cols = [c for s in senders for c in df.columns if (s in c.split(';')[0])]
        elif receivers is not None:
            new_cols = [c for r in receivers for c in df.columns if (r in c.split(';')[1])]
        else:
            new_cols = list(df.columns)
        df = df.reindex(new_cols, fill_value=1.0, axis='columns')
        xlabel = 'Sender-Receiver Pairs'
        ylabel = 'Ligand-Receptor Pairs'
        title = 'Communication Score'
    elif evaluation == 'interactions':
        if senders is not None:
            df = df.reindex(senders, fill_value=0.0, axis='index')
        if receivers is not None:
            df = df.reindex(receivers, fill_value=0.0, axis='columns')
        xlabel = 'Receiver Cells'
        ylabel = 'Sender Cells'
        title = 'CCI Score'

    # Drop all zeros
    df = df.loc[(df != 0).any(axis=1)]
    df = df.T.loc[(df != 0).any(axis=0)].T
    df = pvals[df.columns].loc[df.index].applymap(lambda x: -1. * np.log10(x + 1e-9))

    max_size = mpl.colors.Normalize(vmin=0., vmax=3)

    # Colormap
    cmap = mpl.cm.get_cmap(cmap)

    # Dot plot
    # fig = plt.figure(figsize=figsize)
    fig, (ax2, ax) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 9]})

    for i, idx in enumerate(df.index):
        for j, col in enumerate(df.columns):
            color = np.asarray(cmap(norm(scores[[col]].loc[[idx]].values.item()))).reshape(1, -1)
            size = (max_size(df[[col]].loc[[idx]].values.item()) * tick_size * 2) ** 2
            ax.scatter(j, i, s=size, c=color)
    ax.set_xticks(ticks=range(0, len(df.columns)))
    ax.set_xticklabels(labels=list(df.columns),
                       fontsize=tick_size,
                       rotation=90, ha='center', va='top'
                       )
    ax.set_yticks(ticks=range(0, len(df.index)))
    ax.set_yticklabels(labels=list(df.index),
                       fontsize=tick_size,
                       rotation=0, ha='right', va='center'
                       )

    plt.gca().invert_yaxis()

    plt.tick_params(axis='both',
                    which='both',
                    bottom=True,
                    top=False,
                    right=False,
                    left=True,
                    labelleft=True,
                    labelbottom=True)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    # Colorbar
    # create an axes on the top side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.1 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.1)

    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax,
                        orientation='horizontal'
                        )
    cax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=True
    )  # labels along the bottom edge are off
    cax.set_title(title)

    for i, v in enumerate([np.min(np.min(df)), -1. * np.log10(significance + 1e-9), 3.0]):
        ax2.scatter(i, 0, s=(max_size(v) * tick_size * 2) ** 2, c='k')
        ax2.scatter(i, 1, s=0, c='k')
        if v == 3.0:
            extra = '>='
        else:
            extra = ''
        ax2.annotate(extra+ str(np.round(v, 4)), (i, 0.5))
    ax2.axis('off')
    ax2.set_title('log10(P-value) sizes')

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig
