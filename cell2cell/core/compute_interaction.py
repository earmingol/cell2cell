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
    c1_A = cell1.weighted_ppi['A'].values
    c2_A = cell2.weighted_ppi['A'].values

    numerator = np.nansum(c1_A) + np.nansum(c2_A)
    denominator = np.nansum(c1_A) + np.nansum(c2_A) - numerator


    # NOT USED WITH BIDIRECTIONAL CCI
    # c1_B = cell1.weighted_ppi['B'].values
    # c2_B = cell2.weighted_ppi['B'].values

    # POTENTIAL INTERACTION INDEX USING SCORES OF PROTEIN-PROTEIN INTERACTIONS
    # c1_scores = cell1.weighted_ppi['score'].values
    # c2_scores = cell2.weighted_ppi['score'].values
    # numerator = np.nansum(c1_A * c2_B * c1_scores * c2_scores) + np.nansum(c1_B * c2_A  * c1_scores * c2_scores)
    # denominator = 0.5*(np.nansum(c1_A * c1_scores * c2_scores) + np.nansum(c1_B * c1_scores * c2_scores) +
    #                   np.nansum(c2_A * c1_scores * c2_scores) + np.nansum(c2_B * c1_scores * c2_scores))

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