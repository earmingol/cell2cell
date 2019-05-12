# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


### Pre-process RNAseq data
def drop_empty_genes(rnaseq_data):
    data = rnaseq_data.copy()
    data = data.fillna(0)  # Put zeros to all missing values
    data = data.loc[data.sum(axis=1) != 0]  # Drop rows will 0 among all cell/tissues
    return data


def log10_transformation(rnaseq_data):
    ### Apply this only after applying "drop_empty_genes" function
    data = rnaseq_data.copy()
    data = data.apply(np.log10)
    data = data.replace([np.inf, -np.inf], np.nan)
    return data
