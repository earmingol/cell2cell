import numpy as np
import pandas as pd

from statsmodels.stats.multitest import fdrcorrection


def compute_fdrcorrection_symmetric_matrix(X, alpha=0.1):
    '''
    Computes and FDR correction or Benjamini-Hochberg procedure
    on a symmetric matrix of p-values. Here, only the diagonal
    and values on the upper triangle are considered to avoid
    repetition with the lower triangle.

    Parameters
    ----------
    X : pandas.DataFrame
        A symmetric dataframe of P-values.

    alpha : float, default=0.1
        Error rate of the FDR correction. Must be 0 < alpha < 1.

    Returns
    -------
    adj_X : pandas.DataFrame
        A symmetric dataframe with adjusted P-values of X.
    '''
    pandas = False
    a = X.copy()

    if isinstance(X, pd.DataFrame):
        pandas = True
        a = X.values
        index = X.index
        columns = X.columns

    # Original data
    upper_idx = np.triu_indices_from(a)
    pvals = a[upper_idx]

    # New data
    adj_X = np.zeros(a.shape)
    rej, adj_pvals = fdrcorrection(pvals.flatten(), alpha=alpha)

    # Reorder_data
    adj_X[upper_idx] = adj_pvals
    adj_X = adj_X + np.triu(adj_X, 1).T

    if pandas:
        adj_X = pd.DataFrame(adj_X, index=index, columns=columns)
    return adj_X


def compute_fdrcorrection_asymmetric_matrix(X, alpha=0.1):
    '''
    Computes and FDR correction or Benjamini-Hochberg procedure
    on a asymmetric matrix of p-values. Here, the correction
    is performed for every value in X.

    Parameters
    ----------
    X : pandas.DataFrame
        An asymmetric dataframe of P-values.

    alpha : float, default=0.1
        Error rate of the FDR correction. Must be 0 < alpha < 1.

    Returns
    -------
    adj_X : pandas.DataFrame
        An asymmetric dataframe with adjusted P-values of X.
    '''
    pandas = False
    a = X.copy()

    if isinstance(X, pd.DataFrame):
        pandas = True
        a = X.values
        index = X.index
        columns = X.columns

    # Original data
    pvals = a.flatten()

    # New data
    rej, adj_pvals = fdrcorrection(pvals, alpha=alpha)

    # Reorder_data
    #adj_X = adj_pvals.reshape(-1, a.shape[1])
    adj_X = adj_pvals.reshape(a.shape) # Allows using tensors

    if pandas:
        adj_X = pd.DataFrame(adj_X, index=index, columns=columns)
    return adj_X