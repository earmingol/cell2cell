# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np, pandas as pd


def rnaseq_permutation(rnaseq_data, column='all', seed=None):
    '''This function performs permutations in a given column or all of them (when column='all').'''
    pass

def generate_fake_rnaseq(size, row_names, verbose=True):
    '''
    This function generates a RNA-seq data-set that is genes wise normally distributed, in TPM units (each column).
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

    data = np.random.randn(len(row_names), len(columns))
    min = np.abs(np.amin(data, axis=1))
    min = min.reshape((len(min), 1))

    data = data + min
    if verbose:
        print('Normalizing fake RNA-seq dataset (into TPM)')
    data = 1e6 * np.divide(data, np.sum(data, axis=0))
    df = pd.DataFrame(data, index=row_names, columns=columns)
    df = df.fillna(0)
    #df = 1e6 * df.div(df.sum(axis=0), axis=1)
    return df