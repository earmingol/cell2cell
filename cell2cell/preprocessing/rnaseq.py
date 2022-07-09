# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd


### Pre-process RNAseq datasets
def drop_empty_genes(rnaseq_data):
    '''Drops genes that are all zeroes and/or without
    expression values for all cell-types/tissues/samples.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    Returns
    -------
    data : pandas.DataFrame
        A gene expression data for RNA-seq experiment without
        empty genes. Columns are
        cell-types/tissues/samples and rows are genes.
    '''
    data = rnaseq_data.copy()
    data = data.dropna(how='all')
    data = data.fillna(0)  # Put zeros to all missing values
    data = data.loc[data.sum(axis=1) != 0]  # Drop rows will 0 among all cell/tissues
    return data


def log10_transformation(rnaseq_data, addition = 1e-6):
    '''Log-transforms gene expression values in a
    gene expression matrix for a RNA-seq experiment.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    Returns
    -------
    data : pandas.DataFrame
        A gene expression data for RNA-seq experiment with
        log-transformed values. Values are log10(expression + addition).
        Columns are cell-types/tissues/samples and rows are genes.
    '''
    ### Apply this only after applying "drop_empty_genes" function
    data = rnaseq_data.copy()
    data = data.apply(lambda x: np.log10(x + addition))
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def scale_expression_by_sum(rnaseq_data, axis=0, sum_value=1e6):
    '''
    Normalizes all samples to sum up the same scale factor.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    axis : int, default=0
        Axis to perform the global-scaling normalization. Options
        are {0 for normalizing across rows (column-wise) or 1 for
        normalizing across columns (row-wise)}.

    sum_value : float, default=1e6
        Scaling factor. Normalized axis will sum up this value.

    Returns
    -------
    scaled_data : pandas.DataFrame
        A gene expression data for RNA-seq experiment with
        scaled values. All rows or columns, depending on the specified
        axis sum up to the same value. Columns are
        cell-types/tissues/samples and rows are genes.
    '''
    data = rnaseq_data.values
    data = sum_value * np.divide(data, np.nansum(data, axis=axis))
    scaled_data = pd.DataFrame(data, index=rnaseq_data.index, columns=rnaseq_data.columns)
    return scaled_data


def divide_expression_by_max(rnaseq_data, axis=1):
    '''
    Normalizes each gene value given the max value across an axis.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    axis : int, default=0
        Axis to perform the max-value normalization. Options
        are {0 for normalizing across rows (column-wise) or 1 for
        normalizing across columns (row-wise)}.

    Returns
    -------
    new_data : pandas.DataFrame
        A gene expression data for RNA-seq experiment with
        normalized values. Columns are
        cell-types/tissues/samples and rows are genes.
    '''
    new_data = rnaseq_data.div(rnaseq_data.max(axis=axis), axis=int(not axis))
    new_data = new_data.fillna(0.0).replace(np.inf, 0.0)
    return new_data


def divide_expression_by_mean(rnaseq_data, axis=1):
    '''
    Normalizes each gene value given the mean value across an axis.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    axis : int, default=0
        Axis to perform the mean-value normalization. Options
        are {0 for normalizing across rows (column-wise) or 1 for
        normalizing across columns (row-wise)}.

    Returns
    -------
    new_data : pandas.DataFrame
        A gene expression data for RNA-seq experiment with
        normalized values. Columns are
        cell-types/tissues/samples and rows are genes.
    '''
    new_data = rnaseq_data.div(rnaseq_data.mean(axis=axis), axis=int(not axis))
    new_data = new_data.fillna(0.0).replace(np.inf, 0.0)
    return new_data


def add_complexes_to_expression(rnaseq_data, complexes, agg_method='min'):
    '''
    Adds multimeric complexes into the gene expression matrix.
    Their gene expressions are the minimum expression value
    among the respective subunits composing them.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    complexes : dict
        Dictionary where keys are the complex names in the list of PPIs, while
        values are list of subunits for the respective complex names.

    agg_method : str, default='min'
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.
        - 'gmean' : Geometric mean expression value among all genes.

    Returns
    -------
    tmp_rna : pandas.DataFrame
        Gene expression data for RNA-seq experiment containing multimeric
        complex names. Their gene expressions are the minimum expression value
        among the respective subunits composing them. Columns are
        cell-types/tissues/samples and rows are genes.
    '''
    tmp_rna = rnaseq_data.copy()
    for k, v in complexes.items():
        if all(g in tmp_rna.index for g in v):
            df = tmp_rna.loc[v, :]
            if agg_method == 'min':
                tmp_rna.loc[k] = df.min().values.tolist()
            elif agg_method == 'mean':
                tmp_rna.loc[k] = df.mean().values.tolist()
            elif agg_method == 'gmean':
                tmp_rna.loc[k] = df.apply(lambda x: np.exp(np.mean(np.log(x)))).values.tolist()
            else:
                ValueError("{} is not a valid agg_method".format(agg_method))
        else:
            tmp_rna.loc[k] = [0] * tmp_rna.shape[1]
    return tmp_rna


def aggregate_single_cells(rnaseq_data, metadata, barcode_col='barcodes', celltype_col='cell_types', method='average',
                           transposed=True):
    '''Aggregates gene expression of single cells into cell types for each gene.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a single-cell RNA-seq experiment. Columns are
        single cells and rows are genes. If columns are genes and rows are
        single cells, specify transposed=True.

    metadata : pandas.Dataframe
        Metadata containing the cell types for each single cells in the
        RNA-seq dataset.

    barcode_col : str, default='barcodes'
        Column-name for the single cells in the metadata.

    celltype_col : str, default='cell_types'
        Column-name in the metadata for the grouping single cells into cell types
        by the selected aggregation method.

    method : str, default='average
        Specifies the method to use to aggregate gene expression of single
        cells into their respective cell types. Used to perform the CCI
        analysis since it is on the cell types rather than single cells.
        Options are:

        - 'nn_cell_fraction' : Among the single cells composing a cell type, it
            calculates the fraction of single cells with non-zero count values
            of a given gene.
        - 'average' : Computes the average gene expression among the single cells
            composing a cell type for a given gene.

    transposed : boolean, default=True
        Whether the rnaseq_data is organized with columns as
        genes and rows as single cells.

    Returns
    -------
    agg_df : pandas.DataFrame
        Dataframe containing the gene expression values that were aggregated
        by cell types. Columns are cell types and rows are genes.
    '''
    assert metadata is not None, "Please provide metadata containing the barcodes and cell-type annotation."
    assert method in ['average', 'nn_cell_fraction'], "{} is not a valid option for method".format(method)

    meta = metadata.reset_index()
    meta = meta[[barcode_col, celltype_col]].set_index(barcode_col)
    mapper = meta[celltype_col].to_dict()

    if transposed:
        df = rnaseq_data
    else:
        df = rnaseq_data.T
    df.index = [mapper[c] for c in df.index]
    df.index.name = 'celltype'
    df.reset_index(inplace=True)

    agg_df = pd.DataFrame(index=df.columns).drop('celltype')

    for celltype, ct_df in df.groupby('celltype'):
        ct_df = ct_df.drop('celltype', axis=1)
        if method == 'average':
            agg = ct_df.mean()
        elif method == 'nn_cell_fraction':
            agg = ((ct_df > 0).sum() / ct_df.shape[0])
        agg_df[celltype] = agg
    return agg_df

