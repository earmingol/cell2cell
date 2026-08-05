# -*- coding: utf-8 -*-

from __future__ import absolute_import

import random
import warnings

import numpy as np
import pandas as pd


def check_presence_in_dataframe(df, elements, columns=None):
    '''
    Searches for elements in a dataframe and returns those
    that are present in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe

    elements : list
        List of elements to find in the dataframe. They
        must be a data type contained in the dataframe.

    columns : list, default=None
        Names of columns to consider in the search. If
        None, all columns are used.

    Returns
    -------
    found_elements : list
        List of elements in the input list that were found
        in the dataframe.
    '''
    if columns is None:
        columns = list(df.columns)
    elif isinstance(columns, str):
        # A string would be read by pandas as a single column name, returning a Series
        # instead of the dataframe the code below expects, as in `shuffle_cols_in_df`.
        columns = [columns]
    # `pd.unique` does not sort the values, so it also works when the considered
    # columns contain a mix of data types (e.g. gene names and scores).
    df_elements = pd.Series(pd.unique(df[columns].to_numpy(dtype=object).ravel()))
    found_elements = df_elements.loc[df_elements.isin(elements)].tolist()
    return found_elements


def shuffle_cols_in_df(df, columns, shuffling_number=1, random_state=None):
    '''
    Randomly shuffles specific columns in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe.

    columns : list
        Names of columns to shuffle.

    shuffling_number : int, default=1
        Number of shuffles per column.

    random_state : int, default=None
        Seed for randomization.

    Returns
    -------
    df_ : pandas.DataFrame
        A shuffled dataframe.
    '''
    df_ = df.copy()
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        for i in range(shuffling_number):
            if random_state is not None:
                np.random.seed(random_state + i)
            df_[col] = np.random.permutation(df_[col].values)
    return df_


def shuffle_rows_in_df(df, rows, shuffling_number=1, random_state=None):
    '''
    Randomly shuffles specific rows in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe.

    rows : list
        Names of rows (or indexes) to shuffle.

    shuffling_number : int, default=1
        Number of shuffles per row.

    random_state : int, default=None
        Seed for randomization.

    Returns
    -------
    df_.T : pandas.DataFrame
        A shuffled dataframe.
    '''
    df_ = df.copy().T
    if isinstance(rows, str):
        rows = [rows]

    for row in rows:
        for i in range(shuffling_number):
            if random_state is not None:
                np.random.seed(random_state + i)
            df_[row] = np.random.permutation(df_[row].values)
    return df_.T


def shuffle_dataframe(df, shuffling_number=1, axis=0, random_state=None):
    '''
    Randomly shuffles a whole dataframe across a given axis.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe.

    shuffling_number : int, default=1
        Number of shuffles per column.

    axis : int, default=0
        An axis of the dataframe (0 across rows, 1 across columns).
        Across rows means that shuffles each column independently,
        and across columns shuffles each row independently.

    random_state : int, default=None
        Seed for randomization.

    Returns
    -------
    df_ : pandas.DataFrame
        A shuffled dataframe.
    '''
    df_ = df.copy()
    axis = int(not axis)  # pandas.DataFrame is always 2D
    # `to_numpy(copy=True)` is shuffled in place below, so it must be writable. The array
    # behind `DataFrame.values` is read-only under the copy-on-write of pandas >= 3.0.
    to_shuffle = np.rollaxis(df_.to_numpy(copy=True), axis)
    for _ in range(shuffling_number):
        for i, view in enumerate(to_shuffle):
            if random_state is not None:
                np.random.seed(random_state + i)
            np.random.shuffle(view)
    df_ = pd.DataFrame(np.rollaxis(to_shuffle, axis=axis), index=df_.index, columns=df_.columns)
    return df_


def subsample_dataframe(df, n_samples, random_state=None):
    '''
    Randomly subsamples rows of a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe.

    n_samples : int
        Number of samples, rows in this case. If
        n_samples is larger than the number of rows,
        the entire dataframe will be returned, but
        shuffled.

    random_state : int, default=None
        Seed for randomization.

    Returns
    -------
    subsampled_df : pandas.DataFrame
        A subsampled and shuffled dataframe.
    '''
    items = list(df.index)
    if n_samples > len(items):
        n_samples = len(items)
    if isinstance(random_state, int):
        random.seed(random_state)
    random.shuffle(items)

    subsampled_df = df.loc[items[:n_samples],:]
    return subsampled_df


def check_symmetry(df):
    '''
    Checks whether a dataframe is symmetric.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe.

    Returns
    -------
    symmetric : boolean
        Whether a dataframe is symmetric.
    '''
    shape = df.shape
    if shape[0] == shape[1]:
        symmetric = (df.values.transpose() == df.values).all()
    else:
        symmetric = False
    return symmetric


def zero_diagonal(df):
    '''
    Sets all diagonal elements of a square dataframe to zero.

    Parameters
    ----------
    df : pandas.DataFrame
        A square dataframe.

    Returns
    -------
    df_ : pandas.DataFrame
        A copy of df, but with all diagonal elements with a
        value of zero.
    '''
    # `np.array` always copies, so the result is writable. The array behind
    # `DataFrame.values` is read-only under the copy-on-write of pandas >= 3.0,
    # which makes it unusable with in-place functions such as `np.fill_diagonal`.
    values = np.array(df, dtype=float)
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def convert_to_distance_matrix(df):
    '''
    Converts a symmetric dataframe into a distance dataframe.
    That is, diagonal elements are all zero.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe.

    Returns
    -------
    df_ : pandas.DataFrame
        A copy of df, but with all diagonal elements with a
        value of zero.
    '''
    if check_symmetry(df):
        if np.trace(df.values,) != 0.0:
            # Warned instead of raised, so the diagonal is actually replaced below
            warnings.warn("Diagonal elements are not zero. Automatically replaced by zeros")
        df_ = zero_diagonal(df)
    else:
        raise ValueError('The DataFrame is not symmetric')
    return df_
