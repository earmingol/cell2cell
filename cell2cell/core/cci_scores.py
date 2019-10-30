# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import math

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


def compute_dot_product_score(cell1, cell2, ppi_score=None):
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

    cci_score = np.nansum(c1 * c2 * ppi_score)

    if cci_score is np.nan:
        return 0.0
    return cci_score