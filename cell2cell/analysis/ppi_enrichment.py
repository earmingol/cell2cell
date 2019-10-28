# -*- coding: utf-8 -*-

from __future__ import absolute_import

import itertools
import functools
import numpy as np
import pandas as pd


def ppi_count_operation(pair, subsampled_interactions, use_ppi_score=False, index=None):
    cell1 = pair[0]
    cell2 = pair[1]

    if use_ppi_score:
        assert np.array_equal(subsampled_interactions[0]['ppis'][cell1][['score']].values,
                              subsampled_interactions[0]['ppis'][cell2][['score']].values)

    if index is None:
        if use_ppi_score:
            matrices = [np.multiply(element['ppis'][cell1][['A']].values, element['ppis'][cell2][['B']].values) \
                        * element['ppis'][cell1][['score']].values for element in subsampled_interactions]
        else:
            matrices = [np.multiply(element['ppis'][cell1][['A']].values, element['ppis'][cell2][['B']].values)
                        for element in subsampled_interactions]

    else:
        if use_ppi_score:
            matrices = [
                np.multiply(element['ppis'][cell1][['A']].values[index], element['ppis'][cell2][['B']].values[index]) \
                * element['ppis'][cell1][['score']].values[index] for element in subsampled_interactions]
        else:
            matrices = [np.multiply(element['ppis'][cell1][['A']].values[index], element['ppis'][cell2][['B']].values[index])
                        for element in subsampled_interactions]
    matrices = np.concatenate(matrices, axis=1)
    return matrices


def compute_avg_count_for_ppis(subsampled_interactions, clusters, ppi_data):
    index = ppi_data.apply(lambda row: tuple(sorted([row[0], row[1]])), axis=1)
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

    mean_matrix = mean_matrix.dropna(how='all')
    mean_matrix = mean_matrix.fillna(0.0)
    mean_matrix = mean_matrix.groupby(level=0).sum() / 2.0

    std_matrix = std_matrix.dropna(how='all')
    std_matrix = std_matrix.fillna(0.0)
    std_matrix = std_matrix.groupby(level=0).sum() / 2.0
    return mean_matrix, std_matrix


def get_ppi_score_for_cell_pairs(cells, subsampled_interactions, ppi_data, use_ppi_score=False, ref_ppi_data=None):
    cell_pairs = list(itertools.combinations(cells + cells, 2))
    cell_pairs = list(set(cell_pairs))
    col_labels = ['{};{}'.format(pair[0], pair[1]) for pair in cell_pairs]
    if ref_ppi_data is None:
        index = ppi_data.apply(lambda row: (row[0], row[1]), axis=1)
        keep_index=None
        mean_matrix = pd.DataFrame(index=index, columns=col_labels)
        std_matrix = pd.DataFrame(index=index, columns=col_labels)
    else:
        ref_index = list(ref_ppi_data.apply(lambda row: (row[0], row[1]), axis=1).values)
        keep_index = list(pd.merge(ppi_data, ref_ppi_data, how='inner').index)
        mean_matrix = pd.DataFrame(index=ref_index, columns=col_labels)
        std_matrix = pd.DataFrame(index=ref_index, columns=col_labels)


    for i, pair in enumerate(cell_pairs):
        matrices = ppi_count_operation(pair=pair,
                                       subsampled_interactions=subsampled_interactions,
                                       use_ppi_score=use_ppi_score,
                                       index=keep_index)

        mean_column = np.nanmean(matrices, axis=1)
        std_column = np.nanstd(matrices, axis=1)

        mean_matrix[col_labels[i]] = mean_column
        std_matrix[col_labels[i]] = std_column

    mean_matrix = mean_matrix.dropna(how='all')
    mean_matrix = mean_matrix.fillna(0.0)

    std_matrix = std_matrix.loc[mean_matrix.index, mean_matrix.columns]
    std_matrix = std_matrix.fillna(0.0)
    return mean_matrix, std_matrix