# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


def compute_jaccard_like_cci_score(cell1, cell2, ppi_score=None):
    '''Calculates a Jaccard-like score for the interaction between
    two cells based on their intercellular protein-protein
    interactions such as ligand-receptor interactions.

    Parameters
    ----------
    cell1 : cell2cell.core.cell.Cell
        First cell-type/tissue/sample to compute interaction
        between a pair of them. In a directed interaction,
        this is the sender.

    cell2 : cell2cell.core.cell.Cell
        Second cell-type/tissue/sample to compute interaction
        between a pair of them. In a directed interaction,
        this is the receiver.

    Returns
    -------
    cci_score : float
        Overall score for the interaction between a pair of
        cell-types/tissues/samples. In this case it is a
        Jaccard-like score.
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
    '''Calculates a Bray-Curtis-like score for the interaction between
    two cells based on their intercellular protein-protein
    interactions such as ligand-receptor interactions.

    Parameters
    ----------
    cell1 : cell2cell.core.cell.Cell
        First cell-type/tissue/sample to compute interaction
        between a pair of them. In a directed interaction,
        this is the sender.

    cell2 : cell2cell.core.cell.Cell
        Second cell-type/tissue/sample to compute interaction
        between a pair of them. In a directed interaction,
        this is the receiver.

    Returns
    -------
    cci_score : float
        Overall score for the interaction between a pair of
        cell-types/tissues/samples. In this case is a
        Bray-Curtis-like score.
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
    '''Calculates the number of active protein-protein interactions
    for the interaction between two cells, which could be the number
    of active ligand-receptor interactions.

    Parameters
    ----------
    cell1 : cell2cell.core.cell.Cell
        First cell-type/tissue/sample to compute interaction
        between a pair of them. In a directed interaction,
        this is the sender.

    cell2 : cell2cell.core.cell.Cell
        Second cell-type/tissue/sample to compute interaction
        between a pair of them. In a directed interaction,
        this is the receiver.

    Returns
    -------
    cci_score : float
        Overall score for the interaction between a pair of
        cell-types/tissues/samples.
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


def compute_icellnet_score(cell1, cell2, ppi_score=None):
    '''Calculates the sum of communication scores
    for the interaction between two cells. Based on ICELLNET.

    Parameters
    ----------
    cell1 : cell2cell.core.cell.Cell
        First cell-type/tissue/sample to compute interaction
        between a pair of them. In a directed interaction,
        this is the sender.

    cell2 : cell2cell.core.cell.Cell
        Second cell-type/tissue/sample to compute interaction
        between a pair of them. In a directed interaction,
        this is the receiver.

    Returns
    -------
    cci_score : float
        Overall score for the interaction between a pair of
        cell-types/tissues/samples.
    '''
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    mult = c1 * c2 * ppi_score
    cci_score = np.nansum(mult)

    if cci_score is np.nan:
        return 0.0
    return cci_score


def matmul_jaccard_like(A_scores, B_scores, ppi_score=None):
    '''Computes Jaccard-like scores using matrices of proteins by
    cell-types/tissues/samples.

    Parameters
    ----------
    A_scores : array-like
        Matrix of size NxM, where N are the proteins in the first
        column of a list of PPIs and M are the
        cell-types/tissues/samples.

    B_scores : array-like
        Matrix of size NxM, where N are the proteins in the first
        column of a list of PPIs and M are the
        cell-types/tissues/samples.

    Returns
    -------
    jaccard : numpy.array
        Matrix MxM, representing the CCI score for all pairs of
        cell-types/tissues/samples. In directed interactions,
        the vertical axis (axis 0) represents the senders, while
        the horizontal axis (axis 1) represents the receivers.
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
    '''Computes Bray-Curtis-like scores using matrices of proteins by
    cell-types/tissues/samples.

    Parameters
    ----------
    A_scores : array-like
        Matrix of size NxM, where N are the proteins in the first
        column of a list of PPIs and M are the
        cell-types/tissues/samples.

    B_scores : array-like
        Matrix of size NxM, where N are the proteins in the first
        column of a list of PPIs and M are the
        cell-types/tissues/samples.

    Returns
    -------
    bray_curtis : numpy.array
        Matrix MxM, representing the CCI score for all pairs of
        cell-types/tissues/samples. In directed interactions,
        the vertical axis (axis 0) represents the senders, while
        the horizontal axis (axis 1) represents the receivers.
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
    '''Computes the count of active protein-protein interactions
    used for intercellular communication using matrices of proteins by
    cell-types/tissues/samples.

    Parameters
    ----------
    A_scores : array-like
        Matrix of size NxM, where N are the proteins in the first
        column of a list of PPIs and M are the
        cell-types/tissues/samples.

    B_scores : array-like
        Matrix of size NxM, where N are the proteins in the first
        column of a list of PPIs and M are the
        cell-types/tissues/samples.

    Returns
    -------
    counts : numpy.array
        Matrix MxM, representing the CCI score for all pairs of
        cell-types/tissues/samples. In directed interactions,
        the vertical axis (axis 0) represents the senders, while
        the horizontal axis (axis 1) represents the receivers.
    '''
    if ppi_score is None:
        ppi_score = np.array([1.0] * A_scores.shape[0])
    ppi_score = ppi_score.reshape((len(ppi_score), 1))

    counts = np.matmul(np.multiply(A_scores, ppi_score).transpose(), B_scores)
    return counts


def matmul_cosine(A_scores, B_scores, ppi_score=None):
    '''Computes cosine-similarity scores using matrices of proteins by
    cell-types/tissues/samples.

    Parameters
    ----------
    A_scores : array-like
        Matrix of size NxM, where N are the proteins in the first
        column of a list of PPIs and M are the
        cell-types/tissues/samples.

    B_scores : array-like
        Matrix of size NxM, where N are the proteins in the first
        column of a list of PPIs and M are the
        cell-types/tissues/samples.

    Returns
    -------
    cosine : numpy.array
        Matrix MxM, representing the CCI score for all pairs of
        cell-types/tissues/samples. In directed interactions,
        the vertical axis (axis 0) represents the senders, while
        the horizontal axis (axis 1) represents the receivers.
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