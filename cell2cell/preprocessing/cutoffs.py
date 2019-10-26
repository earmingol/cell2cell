# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.io import read_data

import numpy as np
import pandas as pd


def get_local_percentile_cutoffs(rnaseq_data, percentile=0.75):
    '''
    This function obtains the percentile value across all samples for every gene independently.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Dataframe containing gene expression values for each sample. Genes are rows and samples are columns. Samples could
        correspond to cell/tissue/organ types. The gene names are the index and the sample names the columns of the dataframe.

    percentile : float, 0.75 by default
        This is the percentile to be computed.
    '''
    cutoffs = rnaseq_data.quantile(percentile, axis=1).to_frame()
    cutoffs.columns = ['value']
    return cutoffs


def get_global_percentile_cutoffs(rnaseq_data, percentile=0.75):
    '''
    This function obtains the percentile value across all samples and all genes. The value obtained is used as cutoff
    for all genes.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Dataframe containing gene expression values for each sample. Genes are rows and samples are columns. Samples could
        correspond to cell/tissue/organ types. The gene names are the index and the sample names the columns of the dataframe.

    percentile : float, 0.75 by default
        This is the percentile to be computed.
    '''
    cutoffs = pd.DataFrame(index=rnaseq_data.index, columns=['value'])
    cutoffs['value'] = np.quantile(rnaseq_data.values, percentile)
    return cutoffs


def get_standep_cutoffs(rnaseq_data, k):
    '''Not implemented yet'''
    pass


def get_constant_cutoff(rnaseq_data, constant_cutoff=10):
    '''
    This function generates a cutoff dataframe for all genes in rnaseq_data assigning a constant value.
    '''
    cutoffs = pd.DataFrame(index=rnaseq_data.index)
    cutoffs['value'] = constant_cutoff
    return cutoffs


def get_cutoffs(rnaseq_data, parameters, verbose=True):
    '''
    This function creates cutoff values for genes in rnaseq_data and the respective samples.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Dataframe containing gene expression values for each sample. Genes are rows and samples are columns. Samples could
        correspond to cell/tissue/organ types. The gene names are the index and the sample names the columns of the dataframe.

    parameters : dict
        This dictionary have to contain a 'parameter' key and a 'type' key. The first one is the respective parameter
        associated to the type used.

        Information according to the 'type' used:
            'local_percentile' : computes the value of a given percentile, for each gene independently. In this case,
                                 the parameter corresponds to the percentile to compute, as a float value between 0 and 1.
            'global_percentile' : computes the value of a given percentile from all genes and samples simultaneously.
                                  In this case, the parameter corresponds to the percentile to compute, as a float value
                                  between 0 and 1. All genes have the same cutoff.
            'standep' : Not implemented yet.
            'file' : load a cutoff table from a file. Parameter in this case is the path of that file. It must contain
                     the same genes as index and same samples as columns.
            'multi_col_matrix' : a dataframe must be provided, containing a cutoff for each gene in each sample. This
                                 allows to use specific cutoffs for each sample. The columns here must be the same as the
                                 ones in the rnaseq_data.
            'single_col_matrix' : a dataframe must be provided, containing a cutoff for each gene in only one column. These
                                  cutoffs will be applied to all samples.

    Returns
    -------
    cutoffs : pandas.DataFrame
        Dataframe whose indexes are the same genes that rnaseq_data has as indexes. Depending on the type in the parameters
        dictionary, it may have only one column ('value') or the same columns that rnaseq_data has, generating specfic
        cutoffs for each sample.
    '''
    parameter = parameters['parameter']
    type = parameters['type']
    if verbose:
        print("Calculating cutoffs for gene abundances")
    if type == 'local_percentile':
        cutoffs = get_local_percentile_cutoffs(rnaseq_data, parameter)
        cutoffs.columns = ['value']
    elif type == 'global_percentile':
        cutoffs = get_global_percentile_cutoffs(rnaseq_data, parameter)
        cutoffs.columns = ['value']
    elif type == 'constant_value':
        cutoffs = get_constant_cutoff(rnaseq_data, parameter)
        cutoffs.columns = ['value']
    elif type == 'standep':
        cutoffs = get_standep_cutoffs(rnaseq_data, parameter)
    elif type == 'file':
        cutoffs = read_data.load_cutoffs(parameter,
                                         format='auto')
        cutoffs = cutoffs.loc[rnaseq_data.index]
    elif type == 'multi_col_matrix':
        cutoffs = parameter
        cutoffs = cutoffs.loc[rnaseq_data.index]
        cutoffs = cutoffs[rnaseq_data.columns]
    elif type == 'single_col_matrix':
        cutoffs = parameter
        cutoffs.columns = ['value']
        cutoffs = cutoffs.loc[rnaseq_data.index]
    else:
        raise ValueError(type + ' is not a valid cutoff')
    return cutoffs