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
    c1_A = cell1.binary_ppi['A'].values
    c1_B = cell1.binary_ppi['B'].values

    c2_A = cell2.binary_ppi['A'].values
    c2_B = cell2.binary_ppi['B'].values

    numerator = np.dot(c1_A, c2_B) + np.dot(c1_B, c2_A)
    denominator = 0.5*(np.sum(c1_A) + np.sum(c1_B) + np.sum(c2_A) + np.sum(c2_B))

    if denominator == 0:
        return 0.0

    cci_score = numerator / denominator

    if cci_score is np.nan:
        return 0.0
    return cci_score


def cci_score_from_expression(cell1, cell2):
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

    # NOT IMPLEMENTED YET

    return 0.0