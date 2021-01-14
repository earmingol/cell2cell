# -*- coding: utf-8 -*-

from __future__ import absolute_import

import random
import numpy as np
import pandas as pd


def check_presence_in_dataframe(df, elements, columns=None):
    if columns is None:
        columns = list(df.columns)
    df_elements = pd.Series(np.unique(df[columns].values.flatten()))
    df_elements = df_elements.loc[df_elements.isin(elements)].values
    return list(df_elements)


def shuffle_cols_in_df(df, columns, shuffling_number=1, random_state=None):
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
    df_ = df.copy()
    axis = int(not axis)  # pandas.DataFrame is always 2D
    for _ in range(shuffling_number):
        for i, view in enumerate(np.rollaxis(df_.values, axis)):
            if random_state is not None:
                np.random.seed(random_state + i)
            np.random.shuffle(view)
    return df_


def subsample_dataframe(df, n_samples, random_state=None):
    items = list(df.index)
    if n_samples > len(items):
        n_samples = len(items)
    if isinstance(random_state, int):
        random.seed(random_state)
    random.shuffle(items)

    subsampled_df = df.loc[items[:n_samples],:]
    return subsampled_df


def check_symmetry(df):
    return (df.values.transpose() == df.values).all()


def convert_to_distance_matrix(df):
    if check_symmetry(df):
        df_ = df.copy()
        if np.trace(df_.values,) != 0.0:
            raise Warning("Diagonal elements are not zero. Automatically replaced by zeros")
        np.fill_diagonal(df_.values, 0.0)
    else:
        raise ValueError('The DataFrame is not symmetric')
    return df_
