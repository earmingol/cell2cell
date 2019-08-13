# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd
from cell2cell.preprocessing import rnaseq


def generate_fake_rnaseq(size, row_names, verbose=True):
    '''
    This function generates a RNA-seq datasets-set that is genes wise normally distributed, in TPM units (each column).
    All row values for a given columns sum up to 1 million.

    Parameters
    ----------
    size : int
        Number of samples (columns).

    row_names : array-like
        List containing the name of genes (rows).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing gene expression given the list of genes for each sample.
    '''
    if verbose:
        print('Generating fake RNA-seq dataset.')
    columns = list(range(size))

    data = np.random.randn(len(row_names), len(columns))    # Normal distribution
    min = np.abs(np.amin(data, axis=1))
    min = min.reshape((len(min), 1))

    data = data + min
    if verbose:
        print('Normalizing fake RNA-seq dataset (into TPM)')
    df = rnaseq.scale_expression_by_sum(data, axis=0, sum_value=1e6)
    return df