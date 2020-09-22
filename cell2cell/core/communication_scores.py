# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


def get_binary_scores(cell1, cell2, ppi_score=None):
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    communication_scores = c1 * c2 * ppi_score
    return communication_scores


def get_continuous_scores(cell1, cell2, ppi_score=None):
    # raise ValueError("Continuous communication scores not implemented yet")
    c1 = cell1.weighted_ppi['A'].values
    c2 = cell2.weighted_ppi['B'].values

    if (len(c1) == 0) or (len(c2) == 0):
        return 0.0

    if ppi_score is None:
        ppi_score = np.array([1.0] * len(c1))

    communication_scores = c1 * c2 * ppi_score
    return communication_scores