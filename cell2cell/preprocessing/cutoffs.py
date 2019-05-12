# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd


def percentile_cutoff(rnaseq_data, percentile = 0.8):
    cutoffs = pd.DataFrame()
    cutoffs['cutoff'] = rnaseq_data.quantile(percentile, axis=1)
    cutoffs['gene'] = rnaseq_data['gene_id']
    cutoffs.index = rnaseq_data.index
    return cutoffs


def standep(rnaseq_data, k):
    '''Not implemented yet'''
    pass


def get_cutoffs(rnaseq_data, type ='percentile', verbose = True, **kwargs):
    if verbose:
        print("Calculating cutoffs for gene abundances")
    if type == 'percentile':
        cutoffs = percentile_cutoff(rnaseq_data, kwargs['percentile'])
    elif type == 'standep':
        cutoffs = standep(rnaseq_data, kwargs['k'])
    else:
        raise ValueError(type + ' is not a valid cutoff')
    return cutoffs