# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cell2cell.external import pcoa
from cell2cell.plotting.aesthetics import get_colors_from_labels


def pcoa_3dplot(interaction_space, metadata, sample_col='#SampleID', group_col='Groups', pcoa_method='eigh',
                meta_cmap='gist_rainbow', title='', axis_size=14, legend_size=12, figsize=(6, 5), view_angles=(30, 135),
                filename=None, colors=None, excluded_cells=None):

    if hasattr(interaction_space, 'distance_matrix'):
        print('Interaction space detected as an InteractionSpace class')
        distance_matrix = interaction_space.distance_matrix
    elif (type(interaction_space) is np.ndarray) or (type(interaction_space) is pd.core.frame.DataFrame):
        print('Interaction space detected as a distance matrix')
        distance_matrix = interaction_space
    elif hasattr(interaction_space, 'interaction_space'):
        print('Interaction space detected as a Interactions class')
        distance_matrix = interaction_space.interaction_space.distance_matrix
    else:
        raise ValueError('First run the method compute_pairwise_interactions() in your interaction' + \
                         ' object to generate a distance matrix.')

    # Drop excluded cells
    if excluded_cells is not None:
        df = distance_matrix.loc[~distance_matrix.index.isin(excluded_cells),
                                 ~distance_matrix.columns.isin(excluded_cells)]
    else:
        df = distance_matrix

    # PCoA
    ordination = pcoa(df, method=pcoa_method)
    ordination['samples'].index = df.index

    # Biplot
    fig = plt.figure(figsize=figsize)
    #ax = fig.add_subplot(111, projection='3d')

    ax = Axes3D(fig)

    meta_ = metadata.set_index(sample_col)
    if excluded_cells is not None:
        meta_ = meta_.loc[~meta_.index.isin(excluded_cells)]
    labels = meta_[group_col].values.tolist()

    if colors is None:
        colors = get_colors_from_labels(labels, cmap=meta_cmap)
    else:
        assert all(elem in colors.keys() for elem in set(labels))

    # Plot each data point with respective color
    for i, cell_type in enumerate(sorted(meta_[group_col].unique())):
        cells = list(meta_.loc[meta_[group_col] == cell_type].index)
        if colors is not None:
            ax.scatter(ordination['samples'].loc[cells, 'PC1'],
                       ordination['samples'].loc[cells, 'PC2'],
                       ordination['samples'].loc[cells, 'PC3'],
                       color=colors[cell_type],
                       s=50,
                       edgecolors='k',
                       label=cell_type)
        else:
            ax.scatter(ordination['samples'].loc[cells, 'PC1'],
                       ordination['samples'].loc[cells, 'PC2'],
                       ordination['samples'].loc[cells, 'PC3'],
                       s=50,
                       edgecolors='k',
                       label=cell_type)

    # Plot texts
    ax.set_xlabel('PC1 ({}%)'.format(np.round(ordination['proportion_explained']['PC1'] * 100), 2), fontsize=axis_size)
    ax.set_ylabel('PC2 ({}%)'.format(np.round(ordination['proportion_explained']['PC2'] * 100), 2), fontsize=axis_size)
    ax.set_zlabel('PC3 ({}%)'.format(np.round(ordination['proportion_explained']['PC3'] * 100), 2), fontsize=axis_size)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(view_angles[0], view_angles[1])
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
              ncol=2, fancybox=True, shadow=True, fontsize=legend_size)
    plt.title(title, fontsize=16)

    #distskbio = skbio.DistanceMatrix(df, ids=df.index) # Not using skbio for now

    # Save plot
    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')

    results = {'fig' : fig, 'axes' : ax, 'ordination' : ordination, 'distance_matrix' : df} # df used to be distskbio
    return results