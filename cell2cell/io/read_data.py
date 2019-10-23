# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pickle
import os
import pandas as pd
import numpy as np
from cell2cell.preprocessing import rnaseq, ppi

def load_table(filename, format='auto', sep='\t', sheet_name=False, compression=None):
    '''
    Function to open any table into a pandas dataframe.
    '''
    if filename is None:
        return None

    if format == 'auto':
        if ('.gz' in filename):
            format = 'csv'
            sep = '\t'
            compression='gzip'
        elif ('.xlsx' in filename) or ('.xls' in filename):
            format = 'excel'
        elif ('.csv' in filename):
            format = 'csv'
            sep = ','
            compression=None
        elif ('.tsv' in filename) or ('.txt' in filename):
            format = 'csv'
            sep = '\t'
            compression=None

    if format == 'excel':
        table = pd.read_excel(filename, sheet_name=sheet_name)
    elif (format == 'csv') | (format == 'tsv') | (format == 'txt'):
        table = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        print("Specify a correct format")
        return None
    print(filename + ' was correctly loaded')
    return table


def load_rnaseq(rnaseq_file, gene_column, drop_nangenes=True, log_transformation=False, **kwargs):
    '''
    Load RNAseq datasets from table. Genes names are index. Cells/tissues/organs are columns.





    '''
    print("Opening RNAseq datasets from {}".format(rnaseq_file))
    rnaseq_data = load_table(rnaseq_file, **kwargs)
    if gene_column is None:
        gene_column = list(rnaseq_data.columns)[0]
    rnaseq_data = rnaseq_data.set_index(gene_column)
    # Keep only numeric datasets
    rnaseq_data = rnaseq_data.select_dtypes([np.number])

    if drop_nangenes:
        rnaseq_data = rnaseq.drop_empty_genes(rnaseq_data)

    if log_transformation:
        rnaseq_data = rnaseq.log10_transformation(rnaseq_data)

    rnaseq_data = rnaseq_data.drop_duplicates()
    return rnaseq_data


def load_metadata(metadata_file, rnaseq_data, sample_col=None, **kwargs):
    meta = load_table(metadata_file, **kwargs)
    rnaseq_cols = list(rnaseq_data.columns)
    if sample_col is None:
        sample_col = list(meta.columns)[0]
    meta = meta.set_index(sample_col).loc[rnaseq_cols].reset_index()
    meta.index = [str(i) for i, c in enumerate(rnaseq_data.columns)]
    return meta


def load_cutoffs(cutoff_file, gene_column = None, drop_nangenes = True, log_transformation = False, **kwargs):
    '''
    Load RNAseq datasets from table. Genes names are index. Cells/tissues/organs are columns.





    '''
    print("Opening Cutoff datasets from {}".format(cutoff_file))
    cutoff_data = load_table(cutoff_file, **kwargs)
    if gene_column is not None:
        cutoff_data = cutoff_data.set_index(gene_column)
    else:
        cutoff_data = cutoff_data.set_index(cutoff_data.columns[0])

    # Keep only numeric datasets
    cutoff_data = cutoff_data.select_dtypes([np.number])

    if drop_nangenes:
        cutoff_data = rnaseq.drop_empty_genes(cutoff_data)

    if log_transformation:
        cutoff_data = rnaseq.log10_transformation(cutoff_data)

    cols = list(cutoff_data.columns)
    cols[0] = 'value'
    cutoff_data.columns = cols
    return cutoff_data


def load_ppi(ppi_file, interaction_columns, score=None, rnaseq_genes=None,  **kwargs):
    '''
    Load PPI network from table. Column of Interactor 1 and Interactor 2 must be specified.


    '''
    print("Opening PPI datasets from {}".format(ppi_file))
    ppi_data = load_table(ppi_file,  **kwargs)
    unidirectional_ppi = ppi.remove_ppi_bidirectionality(ppi_data, interaction_columns)
    simplified_ppi = ppi.simplify_ppi(unidirectional_ppi, interaction_columns, score)
    if rnaseq_genes is not None:
        simplified_ppi = ppi.filter_ppi_by_proteins(simplified_ppi, rnaseq_genes)
    return simplified_ppi


def load_go_terms(go_terms_file):
    '''
    Load GO terms obo-basic file
    Example: go_terms_file = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
    '''
    import goenrich

    print("Opening GO terms from {}".format(go_terms_file))
    go_terms = goenrich.obo.ontology(go_terms_file)
    print(go_terms_file + ' was correctly loaded')
    return go_terms


def load_go_annotations(goa_file, experimental_evidence=True):
    '''
    Load GO annotation for a given organism.
    Example: goa_file = 'http://current.geneontology.org/annotations/wb.gaf.gz'
    '''
    import goenrich

    print("Opening GO annotations from {}".format(goa_file))

    goa = goenrich.read.goa(goa_file, experimental_evidence)
    goa_cols = list(goa.columns)
    goa = goa[goa_cols[:3] + [goa_cols[4]]]
    new_cols = ['db', 'Gene', 'Name', 'GO']
    goa.columns = new_cols
    print(goa_file + ' was correctly loaded')
    return goa


def load_variable_with_pickle(filename):
    '''
    Import a large size variable in a python readable way using pickle.
    '''

    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(filename)
    with open(filename, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    variable = pickle.loads(bytes_in)
    return variable