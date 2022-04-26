# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
from scipy.stats.mstats import gmean


def get_binary_scores(cell1, cell2, ppi_score=None):
    '''Computes binary communication scores for all
    protein-protein interactions between a pair of
    cell-types/tissues/samples. This corresponds to
    an AND function between binary values for each
    interacting protein coming from each cell.

    Parameters
    ----------
    cell1 : cell2cell.core.cell.Cell
        First cell-type/tissue/sample to compute the communication
        score. In a directed interaction, this is the sender.

    cell2 : cell2cell.core.cell.Cell
        Second cell-type/tissue/sample to compute the communication
        score. In a directed interaction, this is the receiver.

    ppi_score : array-like, default=None
        An array with a weight for each PPI. The weight
        multiplies the communication scores.

    Returns
    -------
    communication_scores : numpy.array
        An array with the communication scores for each intercellular
        PPI.
    '''
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    communication_scores = c1 * c2 * ppi_score
    return communication_scores


def get_continuous_scores(cell1, cell2, ppi_score=None, method='expression_product'):
    '''Computes continuous communication scores for all
    protein-protein interactions between a pair of
    cell-types/tissues/samples. This corresponds to
    a specific scoring function between preprocessed continuous
    expression values for each interacting protein coming from
    each cell.

    Parameters
    ----------
    cell1 : cell2cell.core.cell.Cell
        First cell-type/tissue/sample to compute the communication
        score. In a directed interaction, this is the sender.

    cell2 : cell2cell.core.cell.Cell
        Second cell-type/tissue/sample to compute the communication
        score. In a directed interaction, this is the receiver.

    ppi_score : array-like, default=None
        An array with a weight for each PPI. The weight
        multiplies the communication scores.

    method : str, default='expression_product'
        Scoring function for computing the communication score.
        Options are:
            - 'expression_product' : Multiplication between the expression
                of the interacting proteins. One coming from cell1 and the
                other from cell2.
            - 'expression_mean' : Average between the expression
                of the interacting proteins. One coming from cell1 and the
                other from cell2.
            - 'expression_gmean' : Geometric mean between the expression
                of the interacting proteins. One coming from cell1 and the
                other from cell2.

    Returns
    -------
    communication_scores : numpy.array
        An array with the communication scores for each intercellular
        PPI.
    '''
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if method == 'expression_product':
        communication_scores = score_expression_product(c1, c2)
    elif method == 'expression_mean':
        communication_scores = score_expression_mean(c1, c2)
    elif method == 'expression_gmean':
        communication_scores = np.sqrt(score_expression_product(c1, c2))
    else:
        raise ValueError('{} is not implemented yet'.format(method))

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    communication_scores = communication_scores * ppi_score
    return communication_scores


def score_expression_product(c1, c2):
    '''Computes the expression product score

    Parameters
    ----------
    c1 : array-like
        A 1D-array containing the preprocessed expression values
        for the interactors in the first column of a list of
        protein-protein interactions.

    c2 : array-like
        A 1D-array containing the preprocessed expression values
        for the interactors in the second column of a list of
        protein-protein interactions.

    Returns
    -------
    c1 * c2 : array-like
        Multiplication of vectors.
    '''
    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0
    return c1 * c2


def score_expression_mean(c1, c2):
    '''Computes the expression product score

    Parameters
    ----------
    c1 : array-like
        A 1D-array containing the preprocessed expression values
        for the interactors in the first column of a list of
        protein-protein interactions.

    c2 : array-like
        A 1D-array containing the preprocessed expression values
        for the interactors in the second column of a list of
        protein-protein interactions.

    Returns
    -------
    (c1 + c2)/2. : array-like
        Average of vectors.
    '''
    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0
    return (c1 + c2)/2.


def compute_ccc_matrix(prot_a_exp, prot_b_exp, communication_score='expression_product'):
    '''Computes communication scores for an specific
    protein-protein interaction using vectors of gene expression
    levels for a given interacting protein produced by
    different cell-types/tissues/samples.

    Parameters
    ----------
    prot_a_exp : array-like
        Vector with gene expression levels for an interacting protein A
        in a given PPI. Coordinates are different cell-types/tissues/samples.

    prot_b_exp : array-like
        Vector with gene expression levels for an interacting protein B
        in a given PPI. Coordinates are different cell-types/tissues/samples.

    communication_score : str, default='expression_product'
        Scoring function for computing the communication score.
        Options are:

        - 'expression_product' : Multiplication between the expression
            of the interacting proteins.
        - 'expression_mean' : Average between the expression
            of the interacting proteins.
        - 'expression_gmean' : Geometric mean between the expression
            of the interacting proteins.

    Returns
    -------
    communication_scores : numpy.array
        Matrix MxM, representing the CCC scores of an specific PPI
        across all pairs of cell-types/tissues/samples. M are all
        cell-types/tissues/samples. In directed interactions, the
        vertical axis (axis 0) represents the senders, while the
        horizontal axis (axis 1) represents the receivers.
    '''
    if communication_score == 'expression_product':
        communication_scores = np.outer(prot_a_exp, prot_b_exp)
    elif communication_score == 'expression_mean':
        communication_scores = (np.outer(prot_a_exp, np.ones(prot_b_exp.shape)) + np.outer(np.ones(prot_a_exp.shape), prot_b_exp)) / 2.
    elif communication_score == 'expression_gmean':
        communication_scores = np.sqrt(np.outer(prot_a_exp, prot_b_exp))
    else:
        raise ValueError("Not a valid communication_score")
    return communication_scores


def aggregate_ccc_matrices(ccc_matrices, method='gmean'):
    '''Aggregates matrices of communication scores. Each
    matrix has the communication scores across all pairs
    of cell-types/tissues/samples for a different
    pair of interacting proteins.

    Parameters
    ----------
    ccc_matrices : list
        List of matrices of communication scores. Each matrix
        is for an specific pair of interacting proteins.

    method : str, default='gmean'.
        Method to aggregate the matrices element-wise.
        Options are:

        - 'gmean' : Geometric mean in an element-wise way.
        - 'sum' : Sum in an element-wise way.
        - 'mean' : Mean in an element-wise way.

    Returns
    -------
    aggregated_ccc_matrix : numpy.array
        A matrix contiaining aggregated communication scores
        from multiple PPIs. It's shape is of MxM, where M are all
        cell-types/tissues/samples. In directed interactions, the
        vertical axis (axis 0) represents the senders, while the
        horizontal axis (axis 1) represents the receivers.
    '''
    if method == 'gmean':
        aggregated_ccc_matrix = gmean(ccc_matrices)
    elif method == 'sum':
        aggregated_ccc_matrix = np.nansum(ccc_matrices, axis=0)
    elif method == 'mean':
        aggregated_ccc_matrix = np.nanmean(ccc_matrices, axis=0)
    else:
        raise ValueError("Not a valid method")

    return aggregated_ccc_matrix