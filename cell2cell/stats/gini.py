# -*- coding: utf-8 -*-

import numpy as np


def gini_coefficient(distribution):
    """Computes the Gini coefficient of an array of values.
    Code borrowed from:
    https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy

    Parameters
    ----------
    distribution : array-like
        An array of values representing the distribution
        to be evaluated.

    Returns
    -------
    gini : float
        Gini coefficient for the evaluated distribution.
    """
    diffsum = 0
    for i, xi in enumerate(distribution[:-1], 1):
        diffsum += np.sum(np.abs(xi - distribution[i:]))
    gini = diffsum / (len(distribution)**2 * np.mean(distribution))
    return gini