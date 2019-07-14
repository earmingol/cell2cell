# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd


### Pre-process RNAseq datasets
def drop_empty_genes(rnaseq_data):
    data = rnaseq_data.copy()
    data = data.fillna(0)  # Put zeros to all missing values
    data = data.loc[data.sum(axis=1) != 0]  # Drop rows will 0 among all cell/tissues
    return data


def log10_transformation(rnaseq_data, addition = 0.65):
    ### Apply this only after applying "drop_empty_genes" function
    data = rnaseq_data.copy()
    data = data.apply(lambda x: np.log10(x + addition))
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def sample_scale_rnaseq_data(rnaseq_data, sum=1e6):
    '''
    This function scale all samples to sum up the same value.
    '''
    data = rnaseq_data.values
    data = sum * np.divide(data, np.sum(data, axis=0))
    scaled_rnaseq_data = pd.DataFrame(data, index=rnaseq_data.index, columns=rnaseq_data.columns)
    return scaled_rnaseq_data


def divide_expression_by_max(rnaseq_data, axis=1):
    '''
    This function divides each gene value given the max value. Axis = 0 is the max of each sample, while axis = 1
    indicates that each gene is divided by its max value across samples.
    '''
    new_data = rnaseq_data.div(rnaseq_data.max(axis=axis), axis=int(not axis))
    return new_data