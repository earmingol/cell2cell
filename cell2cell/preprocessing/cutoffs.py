# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.io import read_data

import numpy as np
import pandas as pd


def get_local_percentile_cutoffs(rnaseq_data, percentile=0.75):
    '''
    Obtains a local value associated with a given percentile across
    cells/tissues/samples for each gene in a rnaseq_data.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment or a single-cell
        experiment after aggregation into cell types. Columns are
        cell-types/tissues/samples and rows are genes.

    percentile : float, default=0.75
        This is the percentile to be computed.

    Returns
    -------
    cutoffs : pandas.DataFrame
        A dataframe containing the value corresponding to the percentile
        across the genes. Rows are genes and the column corresponds to
        'value'.
    '''
    cutoffs = rnaseq_data.quantile(percentile, axis=1).to_frame()
    cutoffs.columns = ['value']
    return cutoffs


def get_global_percentile_cutoffs(rnaseq_data, percentile=0.75):
    '''
    Obtains a global value associated with a given percentile across
    cells/tissues/samples and genes in a rnaseq_data.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment or a single-cell
        experiment after aggregation into cell types. Columns are
        cell-types/tissues/samples and rows are genes.

    percentile : float, default=0.75
        This is the percentile to be computed.

    Returns
    -------
    cutoffs : pandas.DataFrame
        A dataframe containing the value corresponding to the percentile
        across the dataset. Rows are genes and the column corresponds to
        'value'. All values here are the same global percentile.
    '''
    cutoffs = pd.DataFrame(index=rnaseq_data.index, columns=['value'])
    cutoffs['value'] = np.quantile(rnaseq_data.values, percentile)
    return cutoffs


def get_constant_cutoff(rnaseq_data, constant_cutoff=10):
    '''
    Generates a cutoff/threshold dataframe for all genes
    in rnaseq_data assigning a constant value as the cutoff.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment or a single-cell
        experiment after aggregation into cell types. Columns are
        cell-types/tissues/samples and rows are genes.

    constant_cutoff : float, default=10
        Cutoff or threshold assigned to each gene.

    Returns
    -------
    cutoffs : pandas.DataFrame
        A dataframe containing the value corresponding to cutoff or threshold
        assigned to each gene. Rows are genes and the column corresponds to
        'value'. All values are the same and corresponds to the
        constant_cutoff.
    '''
    cutoffs = pd.DataFrame(index=rnaseq_data.index)
    cutoffs['value'] = constant_cutoff
    return cutoffs


def get_cutoffs(rnaseq_data, parameters, verbose=True):
    '''
    This function creates cutoff/threshold values for genes
    in rnaseq_data and the respective cells/tissues/samples
    by a given method or parameter.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment or a single-cell
        experiment after aggregation into cell types. Columns are
        cell-types/tissues/samples and rows are genes.

    parameters : dict
        This dictionary must contain a 'parameter' key and a 'type' key.
        The first one is the respective parameter to compute the threshold
        or cutoff values. The type corresponds to the approach to
        compute the values according to the parameter employed.
        Options of 'type' that can be used:

        - 'local_percentile' : computes the value of a given percentile,
                               for each gene independently. In this case,
                               the parameter corresponds to the percentile
                               to compute, as a float value between 0 and 1.
        - 'global_percentile' : computes the value of a given percentile
                                from all genes and samples simultaneously.
                                In this case, the parameter corresponds to
                                the percentile to compute, as a float value
                                between 0 and 1. All genes have the same cutoff.
        - 'file' : load a cutoff table from a file. Parameter in this case is
                   the path of that file. It must contain the same genes as
                   index and same samples as columns.
        - 'multi_col_matrix' : a dataframe must be provided, containing a
                               cutoff for each gene in each sample. This allows
                               to use specific cutoffs for each sample. The
                               columns here must be the same as the ones in the
                               rnaseq_data.
        - 'single_col_matrix' : a dataframe must be provided, containing a
                                cutoff for each gene in only one column. These
                                cutoffs will be applied to all samples.
        - 'constant_value' : binarizes the expression. Evaluates whether
                             expression is greater than the value input in
                             the 'parameter'.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    cutoffs : pandas.DataFrame
        Dataframe wherein rows are genes in rnaseq_data. Depending on the type in
        the parameters dictionary, it may have only one column ('value') or the
        same columns that rnaseq_data has, generating specfic cutoffs for each
        cell/tissue/sample.
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