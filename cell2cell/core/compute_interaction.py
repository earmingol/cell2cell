# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


def cci_score_from_binary(cell1, cell2):
    '''
    Function that calculates an score for the interaction between two cells based on the interactions of their
    proteins with the proteins of the other cell. This score is based on ........

    Parameters
    ----------
    cell1 : Cell class
        First cell/tissue/organ type to compute interaction between a pair of them.

    cell2 : Cell class
        Second cell/tissue/organ type to compute interaction between a pair of them.

    Returns
    -------
    cci_score : float
        Score for the interaction of the of the pair of cells based on the presence of gene/proteins in the ppi network.
    '''
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    numerator = np.nansum(c1 * c2)
    denominator = np.nansum(c1) + np.nansum(c2) - numerator

    if denominator == 0.0:
        return 0.0

    cci_score = numerator / denominator

    if cci_score is np.nan:
        return 0.0

    return cci_score


def cci_score_from_sampled_weighted(cell1, cell2):
    '''
    Function that calculates an score for the interaction between two cells based on the interactions of their
    proteins with the proteins of the other cell. This score is based on the amount of each protein (expression level)
    rather than its presence or absence (binary level).

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