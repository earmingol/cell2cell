# -*- coding: utf-8 -*-

from __future__ import absolute_import

import itertools
import functools
import numpy as np
import pandas as pd


def compute_avg_count_for_ppis(subsampled_interactions, clusters, ppi_dict):
    interaction_type = subsampled_interactions[0]['interaction_type']
    ppi_network = ppi_dict[interaction_type]

    index = ppi_network.apply(lambda row: tuple(sorted([row[0], row[1]])), axis=1)
    mean_matrix = pd.DataFrame(columns=list(clusters.keys()),
                               index=index)

    std_matrix = pd.DataFrame(columns=list(clusters.keys()),
                               index=index)

    for cluster, cells in clusters.items():
        cell_pairs = list(itertools.combinations(cells + cells, 2))
        cell_pairs = list(set(cell_pairs))  # Remove duplicates

        matrices = list(map(functools.partial(ppi_count_operation, subsampled_interactions=subsampled_interactions),
                            cell_pairs))

        matrices = np.concatenate(matrices, axis=1)
        mean_column = np.nanmean(matrices, axis=1)
        std_column = np.nanstd(matrices, axis=1)

        mean_matrix[cluster] = mean_column
        std_matrix[cluster] = std_column

    mean_matrix = mean_matrix.fillna(0.0)
    mean_matrix = mean_matrix.groupby(level=0).sum() / 2.0

    std_matrix = std_matrix.fillna(0.0)
    std_matrix = std_matrix.groupby(level=0).sum() / 2.0
    return mean_matrix, std_matrix


def ppi_count_operation(pair, subsampled_interactions):
    cell1 = pair[0]
    cell2 = pair[1]

    matrices = [element['ppis'][cell1][['A', 'B']].values * element['ppis'][cell2][['A', 'B']].values
                for element in subsampled_interactions]
    matrices = np.concatenate(matrices, axis=1)
    return matrices
