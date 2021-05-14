# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
from scipy.stats.mstats import gmean


def get_binary_scores(cell1, cell2, ppi_score=None):
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    communication_scores = c1 * c2 * ppi_score
    return communication_scores


def get_continuous_scores(cell1, cell2, ppi_score=None, method='expression_product'):
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if method == 'expression_product':
        communication_scores = score_expression_product(c1, c2)
    elif method == 'expression_mean':
        communication_scores = score_expression_mean(c1, c2)
    else:
        raise ValueError('{} is not implemented yet'.format(method))

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    communication_scores = communication_scores * ppi_score
    return communication_scores


def score_expression_product(c1, c2):
    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0
    return c1 * c2


def score_expression_mean(c1, c2):
    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0
    return (c1 + c2)/2.


def compute_ccc_matrix(prot_a_exp, prot_b_exp, communication_score='expression_product'):
    if communication_score == 'expression_product':
        return np.outer(prot_a_exp, prot_b_exp)
    elif communication_score == 'expression_mean':
        return (np.outer(prot_a_exp, np.ones(prot_b_exp.shape)) + np.outer(np.ones(prot_a_exp.shape), prot_b_exp)) / 2.


def aggregate_ccc_matrices(ccc_matrices, method='gmean'):
    if method == 'gmean':
        aggregated_ccc_matrix = gmean(ccc_matrices)
    elif method == 'sum':
        aggregated_ccc_matrix = np.nansum(ccc_matrices, axis=0)
    elif method == 'mean':
        aggregated_ccc_matrix = np.nanmean(ccc_matrices, axis=0)
    else:
        raise ValueError("Not a valid method")

    return aggregated_ccc_matrix