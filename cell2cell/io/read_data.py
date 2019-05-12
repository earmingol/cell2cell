# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd
import numpy as np
from cell2cell.preprocessing.rnaseq import drop_empty_genes, log10_transformation
from cell2cell.preprocessing.ppi import remove_ppi_bidirectionality, simplify_ppi, ppi_rnaseq_gene_match

def load_table(filename, format ='excel', sep ='\t', sheet_name = False):
    '''Function to open any table into a pandas dataframe.'''
    if format == 'excel':
        table = pd.read_excel(filename, sheet_name = sheet_name)
    elif format == 'csv' or format == 'txt':
        table = pd.read_csv(filename, sep = sep)
    else:
        print("Specify a correct format")
        table = None
    print(filename + ' was correctly loaded')
    return table


def load_rnaseq(rnaseq_file, gene_column, drop_nangenes = True, log_transformation = False, **kwargs):
    '''
    Load RNAseq data from table. Genes names are index. Cells/tissues/organs are columns.





    '''
    print("Opening RNAseq data from {}".format(rnaseq_file))
    rnaseq_data = load_table(rnaseq_file, **kwargs)
    rnaseq_data = rnaseq_data.set_index(gene_column)
    # Keep only numeric data
    rnaseq_data = rnaseq_data.select_dtypes([np.number])

    if drop_nangenes:
        rnaseq_data = drop_empty_genes(rnaseq_data)

    if log_transformation:
        rnaseq_data = log10_transformation(rnaseq_data)

    return rnaseq_data


def load_ppi(ppi_file, interaction_columns, score=None, rnaseq_genes=None,  **kwargs):
    '''Load PPI network from table. Column of Interactor 1 and Interactor 2 must be specified.


    '''
    ppi_data = load_table(ppi_file,  **kwargs)
    unidirectional_ppi = remove_ppi_bidirectionality(ppi_data, interaction_columns)
    simplified_ppi = simplify_ppi(unidirectional_ppi, interaction_columns, score)
    if rnaseq_genes is not None:
        simplified_ppi = ppi_rnaseq_gene_match(simplified_ppi, rnaseq_genes)
    return simplified_ppi


# go_terms_file = 'http://purl.obolibrary.org/obo/go/go-basic.obo'

def load_go_terms(go_terms_file):
    '''Load GO terms obo-basic file'''
    import goenrich

    go_terms = goenrich.obo.ontology(go_terms_file)
    return go_terms


# goa_file = 'http://current.geneontology.org/annotations/wb.gaf.gz'

def load_go_annotations(goa_file):
    '''
    Load GO annotation for a given organism.

    '''
    import goenrich
    goa = goenrich.read.goa(goa_file)
    goa_cols = list(goa.columns)
    goa = goa[goa_cols[:3] + [goa_cols[4]]]
    new_cols = ['db', 'Gene', 'Name', 'GO']
    goa.columns = new_cols
    return goa