# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from cell2cell.plotting.aesthetics import generate_legend, get_colors_from_labels, map_colors_to_metadata


def tensor_factors_plot(interaction_tensor, order_labels=None, metadata=None, sample_col='#SampleID', group_col='Groups',
                        meta_cmaps=None, fontsize=20, plot_legend=True, filename=None):
    assert interaction_tensor.factors is not None, "First run the method 'compute_tensor_factorization' in your InteractionTensor"
    dim = len(interaction_tensor.factors)

    if order_labels is not None:
        assert dim == len(order_labels), "The lenght of factor_labels must match the order of the tensor (order {})".format(dim)
    else:
        order_labels = list(interaction_tensor.factors.keys())

    if metadata is None:
        metadata = [None] * dim
        meta_colors = [None] * dim
        element_colors = [None] * dim
    else:
        if meta_cmaps is None:
            meta_cmaps = ['gist_rainbow']*len(metadata)
        assert len(metadata) == len(meta_cmaps), "Provide a cmap for each metadata element"
        assert len(metadata) == len(interaction_tensor.order_names), "Provide a metadata for each order. If there is no metadata for any, replace with None"
        meta_colors = [get_colors_from_labels(m[group_col], cmap=cmap) if ((m is not None) & (cmap is not None)) else None for m, cmap in zip(metadata, meta_cmaps)]
        element_colors = [map_colors_to_metadata(metadata=m,
                                                 sample_col=sample_col,
                                                 group_col=group_col,
                                                 cmap=cmap).to_dict() if ((m is not None) & (cmap is not None)) else None for m, cmap in zip(metadata, meta_cmaps)]

    rank = interaction_tensor.rank
    factors = interaction_tensor.factors

    # Make the plot
    fig, axes = plt.subplots(nrows=rank,
                             ncols=dim,
                             figsize=(10, int(rank * 1.2 + 1)),
                             sharex='col',
                             #sharey='col'
                             )

    if rank > 1:
        # Iterates horizontally
        count = 0
        for ind, (order_factors, axs) in enumerate(zip(factors.values(), axes.T)):
            # Iterates vertically
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


def plot_elbow(loss, figsize=(4, 2.25), fontsize=14, filename=None):

    fig = plt.figure(figsize=figsize)

    plt.plot(*zip(*loss))
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.xlabel('Rank', fontsize=int(1.2*fontsize))
    plt.ylabel('Error', fontsize=int(1.2 * fontsize))

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return fig


def plot_multiple_run_elbow(all_loss, figsize=(4, 2.25), fontsize=14, filename=None):

    fig = plt.figure(figsize=figsize)

    x = list(range(1, all_loss.shape[1]+1))
    mean = np.nanmean(all_loss, axis=0)
    std = np.nanstd(all_loss, axis=0)

    # Plot Mean
    plt.plot(x, mean, 'ob')

    # Plot Std
    plt.fill_between(x, mean-std, mean+std, color='steelblue', alpha=.2,
                     label='$\pm$ 1 std')


    plt.tick_params(axis='both', labelsize=fontsize)
    plt.xlabel('Rank', fontsize=int(1.2*fontsize))
    plt.ylabel('Error', fontsize=int(1.2 * fontsize))

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return fig

def generate_plot_df(interaction_tensor):
    factor_labels = ['Time', 'LRs', 'Sender', 'Receiver']
    plot_df = pd.DataFrame()
    for lab, order_factors in enumerate(interaction_tensor.factors.values()):
        sns_df = order_factors.T
        sns_df.index.name = 'Factors'
        melt_df = pd.melt(sns_df.reset_index(), id_vars=['Factors'], value_vars=sns_df.columns)
        melt_df = melt_df.assign(Order=factor_labels[lab])

        plot_df = pd.concat([plot_df, melt_df])
    plot_df.columns = ['Factor', 'Variable', 'Value', 'Order']

    return plot_df