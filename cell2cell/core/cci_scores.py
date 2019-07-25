# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


def compute_jaccard_like_cci_score(cell1, cell2):
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

    # Extended Jaccard similarity
    numerator = np.nansum(c1 * c2)
    denominator = np.nansum(c1 * c1) + np.nansum(c2 * c2) - numerator

    if denominator == 0.0:
        return 0.0

    cci_score = numerator / denominator

    if cci_score is np.nan:
        return 0.0
    return cci_score


def compute_dice_cci_score(cell1, cell2):
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
    from scipy.spatial.distance import dice
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    cci_score = 1.0 - dice(c1, c2)

    if cci_score is np.nan:
        return 0.0
    return cci_score


def compute_braycurtis_cci_score(cell1, cell2):
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
    from scipy.spatial.distance import braycurtis
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    cci_score = 1.0 - braycurtis(c1, c2)

    if cci_score is np.nan:
        return 0.0
    return cci_score


def compute_correlation_cci_score(cell1, cell2):
    '''
    This function
    '''
    from scipy.spatial.distance import correlation

    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    cci_score = 1.0 - correlation(c1, c2)

    if cci_score is np.nan:
        return 0.0
    return cci_score