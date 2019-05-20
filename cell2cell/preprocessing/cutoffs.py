# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.io import read_data

import numpy as np
import pandas as pd


def percentile_cutoff(rnaseq_data, percentile = 0.8):
    cutoffs = rnaseq_data.quantile(percentile, axis=1).to_frame()
    #cutoffs.index = rnaseq_data.index
    cutoffs.columns = ['value']
    return cutoffs


def standep(rnaseq_data, k):
    '''Not implemented yet'''
    pass


def get_cutoffs(rnaseq_data, parameters, verbose=True):
    parameter = parameters['parameter']
    type = parameters['type']
    if verbose:
        print("Calculating cutoffs for gene abundances")
    if type == 'percentile':
        cutoffs = percentile_cutoff(rnaseq_data, parameter)
    elif type == 'standep':
        cutoffs = standep(rnaseq_data, parameter)
    elif type == 'file':
        if parameter['file'] is not None:
            cutoffs = read_data.load_cutoffs(parameters['file'],
                                             drop_nangenes=True,
                                             log_transformation=False,
                                             format='excel')
        else:
            raise ValueError("Please provide complete path for the cutoff file (assign it to the key 'file' in parameters)")
    else:
        raise ValueError(type + ' is not a valid cutoff')
    cutoffs.columns = ['value']
    return cutoffs