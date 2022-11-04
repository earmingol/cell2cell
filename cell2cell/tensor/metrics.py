# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from itertools import combinations

# Authors: Hratch Baghdassarian <hmbaghdassarian@gmail.com>, Erick Armingol <earmingol14@gmail.com>
# similarity metrics for tensor decompositions


def correlation_index(factors_1, factors_2, tol=5e-16, method='stacked'):
    """
    CorrIndex implementation to assess tensor decomposition outputs.
    From [1] Sobhani et al 2022 (https://doi.org/10.1016/j.sigpro.2022.108457).
    Metric is scaling and column-permutation invariant, wherein each column is a factor.

    Parameters
    ----------
    factors_1 : dict
        Ordered dictionary containing a dataframe with the factor loadings for each
        dimension/order of the tensor. This is the result from a tensor decomposition,
        it can be found as the attribute `factors` in any tensor class derived from the
        class BaseTensor (e.g. BaseTensor.factors).

    factors_2 : dict
        Similar to factors_1 but coming from another tensor decomposition of a tensor
        with equal shape.

    tol : float, default=5e-16
        Precision threshold below which to call the CorrIndex score 0.

    method : str, default='stacked'
        Method to obtain the CorrIndex by comparing the A matrices from two decompositions.
        Possible options are:

        - 'stacked' : The original method implemented in [1]. Here all A matrices from the same decomposition are
                      vertically concatenated, building a big A matrix for each decomposition.
        - 'max_score' : This computes the CorrIndex for each pair of A matrices (i.e. between A_1 in factors_1 and
                        factors_2, between A_2 in factors_1 and factors_2, and so on). Then the max score is
                        selected (the most conservative approach). In other words, it selects the max score among the
                        CorrIndexes computed dimension-wise.
        - 'min_score' : Similar to 'max_score', but the min score is selected (the least conservative approach).
        - 'avg_score' : Similar to 'max_score', but the avg score is selected.

    Returns
    -------
    score : float
         CorrIndex metric [0,1]; lower score indicates higher similarity between matrices
    """
    factors_1 = list(factors_1.values())
    factors_2 = list(factors_2.values())

    # check input factors shape
    for factors in [factors_1, factors_2]:
        if len({np.shape(A)[1]for A in factors}) != 1:
            raise ValueError('Factors should be a list of loading matrices of the same rank')

    # check method
    options = ['stacked', 'max_score', 'min_score', 'avg_score']
    if method not in options:
        raise ValueError("The `method` must be either option among {}".format(options))

    if method == 'stacked':
        # vertically stack loading matrices -- shape sum(tensor.shape)xR)
        X_1 = [np.concatenate(factors_1, 0)]
        X_2 = [np.concatenate(factors_2, 0)]
    else:
        X_1 = factors_1
        X_2 = factors_2

    for x1, x2 in zip(X_1, X_2):
        if np.shape(x1) != np.shape(x2):
            raise ValueError('Factor matrices should be of the same shapes')

    # normalize columns to L2 norm - even if ran decomposition with normalize_factors=True
    col_norm_1 = [np.linalg.norm(x1, axis=0) for x1 in X_1]
    col_norm_2 = [np.linalg.norm(x2, axis=0) for x2 in X_2]
    for cn1, cn2 in zip(col_norm_1, col_norm_2):
        if np.any(cn1 == 0) or np.any(cn2 == 0):
            raise ValueError('Column norms must be non-zero')
    X_1 = [x1 / cn1 for x1, cn1 in zip(X_1, col_norm_1)]
    X_2 = [x2 / cn2 for x2, cn2 in zip(X_2, col_norm_2)]

    corr_idxs = [_compute_correlation_index(x1, x2, tol=tol) for x1, x2 in zip(X_1, X_2)]

    if method == 'stacked':
        score = corr_idxs[0]
    elif method == 'max_score':
        score = np.max(corr_idxs)
    elif method == 'min_score':
        score = np.min(corr_idxs)
    elif method == 'avg_score':
        score = np.mean(corr_idxs)
    else:
        score = 1.0
    return score


def _compute_correlation_index(x1, x2, tol=5e-16):
    '''
    Computes the CorrIndex from the L2-normalized A matrices.

    Parameters
    ----------
    x1 : list
        A list containing normalized A matrix(ces) from the first tensor decomposition.

    x2 : list
        A list containing normalized A matrix(ces) from the first tensor decomposition.

    tol : float, default=5e-16
        Precision threshold below which to call the CorrIndex score 0, by default 5e-16

    Returns
    -------
    score : float
         CorrIndex metric [0,1]; lower score indicates higher similarity between matrices
    '''
    # generate the correlation index input
    c_prod_mtx = np.abs(np.matmul(np.conj(np.transpose(np.asarray(x1))), np.asarray(x2)))

    # correlation index scoring
    n_elements = np.shape(c_prod_mtx)[1] + np.shape(c_prod_mtx)[0]
    score = (1 / (n_elements)) * (np.sum(np.abs(np.max(c_prod_mtx, 1) - 1)) + np.sum(np.abs(np.max(c_prod_mtx, 0) - 1)))
    if score < tol:
        score = 0
    return score


def pairwise_correlation_index(factors, tol=5e-16, method='stacked'):
    '''
    Computes the CorrIndex between all pairs of factors

    Parameters
    ----------
    factors : list
        List with multiple Ordered dictionaries, each containing a dataframe with
        the factor loadings for each dimension/order of the tensor. This is the
        result from a tensor decomposition, it can be found as the attribute
        `factors` in any tensor class derived from the class BaseTensor
        (e.g. BaseTensor.factors).

    tol : float, default=5e-16
        Precision threshold below which to call the CorrIndex score 0.

    method : str, default='stacked'
        Method to obtain the CorrIndex by comparing the A matrices from two decompositions.
        Possible options are:

        - 'stacked' : The original method implemented in [1]. Here all A matrices from the same decomposition are
                      vertically concatenated, building a big A matrix for each decomposition.
        - 'max_score' : This computes the CorrIndex for each pair of A matrices (i.e. between A_1 in factors_1 and
                        factors_2, between A_2 in factors_1 and factors_2, and so on). Then the max score is
                        selected (the most conservative approach). In other words, it selects the max score among the
                        CorrIndexes computed dimension-wise.
        - 'min_score' : Similar to 'max_score', but the min score is selected (the least conservative approach).
        - 'avg_score' : Similar to 'max_score', but the avg score is selected.

    Returns
    -------
    scores : pd.DataFrame
         Dataframe with CorrIndex metric for each pair of decompositions.
         This metric bounds are [0,1]; lower score indicates higher similarity between matrices
    '''
    N = len(factors)
    idxs = list(range(N))
    pairs = list(combinations(idxs, 2))
    scores = pd.DataFrame(np.zeros((N, N)),index=idxs, columns=idxs)
    for p1, p2 in pairs:
        corrindex = correlation_index(factors_1=factors[p1],
                                      factors_2=factors[p2],
                                      tol=tol,
                                      method=method
                                      )

        scores.at[p1, p2] = corrindex
        scores.at[p2, p1] = corrindex
    return scores
