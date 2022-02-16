# -*- coding: utf-8 -*-

import numpy as np


def normalize_factors(factors):
    '''
    L2-normalizes the factors considering all tensor dimensions
    from a tensor decomposition result

    Parameters
    ----------
    factors : dict
        Ordered dictionary containing a dataframe with the factor loadings for each
        dimension/order of the tensor. This is the result from a tensor decomposition,
        it can be found as the attribute `factors` in any tensor class derived from the
        class BaseTensor (e.g. BaseTensor.factors).

    Returns
    -------
    norm_factors : dict
        The normalized factors.
    '''
    norm_factors = dict()
    for k, v in factors.items():
        norm_factors[k] = v / np.linalg.norm(v, axis=0)
    return norm_factors


def shuffle_factors(factors, axis=0):
    '''
    Randomly shuffles the values of the factors in the tensor decomposition.
    '''
    raise NotImplementedError
