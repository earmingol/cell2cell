import numpy as np
import pandas as pd

from statsmodels.stats.multitest import fdrcorrection


def compute_fdrcorrection_symmetric_matrix(X, alpha=0.1):
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