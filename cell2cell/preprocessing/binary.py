# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd


def get_binary_rnaseq(rnaseq_data, cutoffs):
    binary_rnaseq_data = rnaseq_data.copy()
    for column in rnaseq_data.columns:
        binary_rnaseq_data[column] = binary_rnaseq_data[column].to_frame().apply(axis=1, func=lambda row: 1 if (row[column] >= cutoffs.loc[row.name, 'value']) else 0).values
    return binary_rnaseq_data


def get_binary_ppi(ppi_data, binary_rnaseq_data, column='value'):
    binary_ppi = pd.DataFrame(columns=ppi_data.columns)
    binary_ppi['A'] = ppi_data['A'].apply(func=lambda row: binary_rnaseq_data.loc[row, column])
    binary_ppi['B'] = ppi_data['B'].apply(func=lambda row: binary_rnaseq_data.loc[row, column])
    binary_ppi = binary_ppi[['A', 'B']].reset_index(drop=True)
    return binary_ppi