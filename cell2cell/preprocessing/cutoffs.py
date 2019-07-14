# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.io import read_data

import numpy as np
import pandas as pd


def get_local_percentile_cutoffs(rnaseq_data, percentile = 0.8):
    cutoffs = rnaseq_data.quantile(percentile, axis=1).to_frame()
    cutoffs.columns = ['value']
    return cutoffs

def get_global_percentile_cutoffs(rnaseq_data, percentile = 0.8):
    cutoffs = pd.DataFrame(index=rnaseq_data.index, columns=['value'])
    cutoffs['value'] = np.quantile(rnaseq_data.values, percentile)
    return cutoffs

def get_standep_cutoffs(rnaseq_data, k):
    '''Not implemented yet'''
    pass


def get_cutoffs(rnaseq_data, parameters, verbose=True):
    parameter = parameters['parameter']
    type = parameters['type']
    if verbose:
        print("Calculating cutoffs for gene abundances")
    if type == 'local_percentile':
        cutoffs = get_local_percentile_cutoffs(rnaseq_data, parameter)
    elif type == 'global_percentile':
        cutoffs = get_global_percentile_cutoffs(rnaseq_data, parameter)
    elif type == 'standep':
        cutoffs = get_standep_cutoffs(rnaseq_data, parameter)
    elif type == 'file':
        cutoffs = read_data.load_cutoffs(parameter,
                                         format='auto')
    else:
        raise ValueError(type + ' is not a valid cutoff')
    cutoffs.columns = ['value']
    return cutoffs