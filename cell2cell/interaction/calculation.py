# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


def compute_cci_score(cell1, cell2):
    '''Function that calculate an score for the interaction between two cells based on the interactions of their proteins
    with the proteins of the other cell.
    This score is based on ........

    Parameters
    ----------
    cell1 : Cell class


    cell2 : Cell class

    Returns
    -------
    cci_score : float
        Score for the interaction of the gene/protein of the pair of cells based on their relative abundances.
    '''
    c1_A = cell1.binary_ppi['A'].values
    c1_B = cell1.binary_ppi['B'].values

    c2_A = cell2.binary_ppi['A'].values
    c2_B = cell2.binary_ppi['B'].values

    numerator = np.dot(c1_A, c2_B) + np.dot(c1_B, c2_A)
    denominator = 0.5*(np.sum(c1_A) + np.sum(c1_B) + np.sum(c2_A) + np.sum(c2_B))

    if denominator == 0:
        return 0

    cci_score = numerator / denominator

    if cci_score is np.nan:
        return 0.0
    return cci_score