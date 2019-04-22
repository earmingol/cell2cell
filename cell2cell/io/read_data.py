# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd
from cell2cell.preprocessing.data_manipulation import drop_empty_genes, log10_transformation

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


def load_rnaseq(rnaseq_file, cell_columns, gene_column, drop_nangenes = True, log_transformation = False, **kwargs):
    '''
    Load RNAseq data from table. Genes names are index. Cells/tissues/organs are columns.





    '''
    print("Opening RNAseq data from {}".format(rnaseq_file))
    rnaseq_data = load_table(rnaseq_file, **kwargs)
    rnaseq_data = rnaseq_data.set_index(gene_column)
    rnaseq_data = rnaseq_data[cell_columns]

    if drop_nangenes:
        rnaseq_data = drop_empty_genes(rnaseq_data)

    if log_transformation:
        rnaseq_data = log10_transformation(rnaseq_data)

    return rnaseq_data


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