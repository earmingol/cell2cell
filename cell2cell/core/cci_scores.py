# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


def compute_jaccard_like_cci_score(cell1, cell2, ppi_score=None):
    '''
    Function that calculates an extended jaccard-like score for the interaction between two cells based on the
    interactions of their proteins with the proteins of the other cell.

    Parameters
    ----------
    cell1 : Cell class
        First cell/tissue/organ type to compute interaction between a pair of them.

    cell2 : Cell class
        Second cell/tissue/organ type to compute interaction between a pair of them.

    Returns
    -------
    cci_score : float
        Score for the interaction of the of the pair of cells based on the abundance of gene/proteins in the ppi network.
    '''
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    # Extended Jaccard similarity
    numerator = np.nansum(c1 * c2 * ppi_score)
    denominator = np.nansum(c1 * c1 * ppi_score) + np.nansum(c2 * c2 * ppi_score) - numerator

    if denominator == 0.0:
        return 0.0

    cci_score = numerator / denominator

    if cci_score is np.nan:
        return 0.0
    return cci_score


def compute_braycurtis_like_cci_score(cell1, cell2, ppi_score=None):
    '''
    Function that calculates an extended bray-curtis-like score for the interaction between two cells based on the
    interactions of their proteins with the proteins of the other cell.

    Parameters
    ----------
    cell1 : Cell class
        First cell/tissue/organ type to compute interaction between a pair of them.

    cell2 : Cell class
        Second cell/tissue/organ type to compute interaction between a pair of them.

    Returns
    -------
    cci_score : float
        Score for the interaction of the of the pair of cells based on the abundance of gene/proteins in the ppi network.
    '''
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    # Bray Curtis similarity
    numerator = 2 * np.nansum(c1 * c2 * ppi_score)
    denominator = np.nansum(c1 * c1 * ppi_score) + np.nansum(c2 * c2 * ppi_score)

    if denominator == 0.0:
        return 0.0

    cci_score = numerator / denominator

    if cci_score is np.nan:
        return 0.0
    return cci_score


def compute_count_score(cell1, cell2, ppi_score=None):
    '''
    Function that calculates an dot score for the interaction between two cells based on the
    interactions of their proteins with the proteins of the other cell.

    Parameters
    ----------
    cell1 : Cell class
        First cell/tissue/organ type to compute interaction between a pair of them.

    cell2 : Cell class
        Second cell/tissue/organ type to compute interaction between a pair of them.

    Returns
    -------
    cci_score : float
        Score for the interaction of the of the pair of cells based on the abundance of gene/proteins in the ppi network.
    '''
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    mult = c1 * c2 * ppi_score
    cci_score = np.nansum(mult != 0) # Count all active pathways (different to zero)

    if cci_score is np.nan:
        return 0.0
    return cci_score


def matmul_jaccard_like(A_scores, B_scores, ppi_score=None):
    '''All cell-pair scores from two matrices. A_scores contains the communication scores for the partners on the left
     column of the PPI network and B_score contains the same but for the partners in the right column.

     Parameters
     __________
     A_scores : array
        Matrix of size NxM, where N is the number of PPIs and M is the number of individual cells.

    B_scores : array
        Matrix of size NxM, where N is the number of PPIs and M is the number of individual cells.

    Returns
    -------
    jaccard : array
        Matrix MxM, representing the CCI score for all cell pairs
    '''
    if ppi_score is None:
        ppi_score = np.array([1.0] * A_scores.shape[0])
    ppi_score = ppi_score.reshape((len(ppi_score), 1))

    numerator = np.matmul(np.multiply(A_scores, ppi_score).transpose(), B_scores)

    A_module = np.sum(np.multiply(np.multiply(A_scores, A_scores), ppi_score), axis=0)
    B_module = np.sum(np.multiply(np.multiply(B_scores, B_scores), ppi_score), axis=0)
    denominator = A_module.reshape((A_module.shape[0], 1)) + B_module - numerator

    jaccard = np.divide(numerator, denominator)
    return jaccard


def matmul_bray_curtis_like(A_scores, B_scores, ppi_score=None):
    '''All cell-pair scores from two matrices. A_scores contains the communication scores for the partners on the left
     column of the PPI network and B_score contains the same but for the partners in the right column.

     Parameters
     __________
     A_scores : array
        Matrix of size NxM, where N is the number of PPIs and M is the number of individual cells.

    B_scores : array
        Matrix of size NxM, where N is the number of PPIs and M is the number of individual cells.

    Returns
    -------
    bray_curtis : array
        Matrix MxM, representing the CCI score for all cell pairs
    '''
    if ppi_score is None:
        ppi_score = np.array([1.0] * A_scores.shape[0])
    ppi_score = ppi_score.reshape((len(ppi_score), 1))

    numerator = np.matmul(np.multiply(A_scores, ppi_score).transpose(), B_scores)

    A_module = np.sum(np.multiply(np.multiply(A_scores, A_scores), ppi_score), axis=0)
    B_module = np.sum(np.multiply(np.multiply(B_scores, B_scores), ppi_score), axis=0)
    denominator = A_module.reshape((A_module.shape[0], 1)) + B_module

    bray_curtis = np.divide(2.0*numerator, denominator)
    return bray_curtis


def matmul_count_active(A_scores, B_scores, ppi_score=None):
    '''All cell-pair scores from two matrices. A_scores contains the communication scores for the partners on the left
     column of the PPI network and B_score contains the same but for the partners in the right column.

     Parameters
     __________
     A_scores : array
        Matrix of size NxM, where N is the number of PPIs and M is the number of individual cells.

    B_scores : array
        Matrix of size NxM, where N is the number of PPIs and M is the number of individual cells.

    Returns
    -------
    counts : array
        Matrix MxM, representing the CCI score for all cell pairs
    '''
    if ppi_score is None:
        ppi_score = np.array([1.0] * A_scores.shape[0])
    ppi_score = ppi_score.reshape((len(ppi_score), 1))

    counts = np.matmul(np.multiply(A_scores, ppi_score).transpose(), B_scores)
    return counts


def matmul_cosine(A_scores, B_scores, ppi_score=None):
    '''All cell-pair scores from two matrices. A_scores contains the communication scores for the partners on the left
     column of the PPI network and B_score contains the same but for the partners in the right column.

     Parameters
     __________
     A_scores : array
        Matrix of size NxM, where N is the number of PPIs and M is the number of individual cells.

    B_scores : array
        Matrix of size NxM, where N is the number of PPIs and M is the number of individual cells.

    Returns
    -------
    counts : array
        Matrix MxM, representing the CCI score for all cell pairs
    '''
    if ppi_score is None:
        ppi_score = np.array([1.0] * A_scores.shape[0])
    ppi_score = ppi_score.reshape((len(ppi_score), 1))

    numerator = np.matmul(np.multiply(A_scores, ppi_score).transpose(), B_scores)

    A_module = np.sum(np.multiply(np.multiply(A_scores, A_scores), ppi_score), axis=0) ** 0.5
    B_module = np.sum(np.multiply(np.multiply(B_scores, B_scores), ppi_score), axis=0) ** 0.5
    denominator = A_module.reshape((A_module.shape[0], 1)) * B_module

    cosine = np.divide(numerator, denominator)
    return cosine