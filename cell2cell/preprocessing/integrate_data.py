# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd
import cell2cell.preprocessing as preprocessing
from cell2cell.preprocessing import ppi, gene_ontology

## RNAseq datasets
def get_modified_rnaseq(rnaseq_data, score_type='binary', **kwargs):
    if score_type == 'binary':
        modified_rnaseq = get_binary_rnaseq(rnaseq_data, kwargs['cutoffs'])
    elif score_type == 'weighted':
        modified_rnaseq = preprocessing.divide_expression_by_max(rnaseq_data, axis=1)
    else:
        # As other score types are implemented, other elif condition will be included here.
        raise NotImplementedError("Score type {} to compute pairwise cell-interactions is not implemented".format(score_type))
    return modified_rnaseq


def get_binary_rnaseq(rnaseq_data, cutoffs):
    binary_rnaseq_data = rnaseq_data.copy()
    binary_rnaseq_data = binary_rnaseq_data.ge(list(cutoffs.value.values), axis=0)
    binary_rnaseq_data= binary_rnaseq_data.astype(float)
    return binary_rnaseq_data


## PPI datasets
def get_weighted_ppi(ppi_data, modified_rnaseq_data, column='value', score_type='binary'):
    if score_type == 'binary':
        weighted_ppi = get_binary_ppi(ppi_data, modified_rnaseq_data, column)
    elif score_type == 'weighted':
        weighted_ppi = get_sample_weighted_ppi(ppi_data, modified_rnaseq_data, column)
    else:
        # As other score types are implemented, other elif condition will be included here.
        raise NotImplementedError(
            "Score type {} to compute pairwise cell-interactions is not implemented".format(score_type))
    return weighted_ppi


def get_binary_ppi(ppi_data, binary_rnaseq_data, column='value'):
    binary_ppi = ppi_data.copy()
    binary_ppi['A'] = binary_ppi['A'].apply(func=lambda row: binary_rnaseq_data.loc[row, column])
    binary_ppi['B'] = binary_ppi['B'].apply(func=lambda row: binary_rnaseq_data.loc[row, column])
    binary_ppi = binary_ppi[['A', 'B', 'score']].reset_index(drop=True)
    return binary_ppi


def get_sample_weighted_ppi(ppi_data, modified_rnaseq_data, column='value'):
    sample_weighted_ppi = ppi_data.copy()
    sample_weighted_ppi['A'] = sample_weighted_ppi['A'].apply(func=lambda row: modified_rnaseq_data.loc[row, column])
    sample_weighted_ppi['B'] = sample_weighted_ppi['B'].apply(func=lambda row: modified_rnaseq_data.loc[row, column])
    sample_weighted_ppi = sample_weighted_ppi[['A', 'B', 'score']].reset_index(drop=True)
    return sample_weighted_ppi


def get_ppi_dict_from_proteins(ppi_data, contact_proteins, mediator_proteins=None, interaction_columns=['A', 'B']):

    ppi_dict = dict()
    ppi_dict['contacts'] = ppi.filter_ppi_network(ppi_data,
                                              contact_proteins,
                                              mediator_proteins=mediator_proteins,
                                              interaction_type='contacts',
                                              interaction_columns=interaction_columns)
    if mediator_proteins is not None:
        ppi_dict['mediated'] = ppi.filter_ppi_network(ppi_data,
                                                      contact_proteins,
                                                      mediator_proteins=mediator_proteins,
                                                      interaction_type='mediated',
                                                      interaction_columns=interaction_columns)

        ppi_dict['combined'] = ppi.filter_ppi_network(ppi_data,
                                                      contact_proteins,
                                                      mediator_proteins=mediator_proteins,
                                                      interaction_type='combined',
                                                      interaction_columns=interaction_columns)
    return ppi_dict


def get_ppi_dict_from_go_terms(ppi_data, go_annotations, go_terms, contact_go_terms, mediator_go_terms=None, use_children=True,
                               go_header='GO', gene_header='Gene', interaction_columns=['A', 'B'], verbose=True):

    if use_children == True:
        contact_proteins = gene_ontology.get_genes_from_go_hierarchy(go_annotations,
                                                                     go_terms,
                                                                     contact_go_terms,
                                                                     go_header=go_header,
                                                                     gene_header=gene_header,
                                                                     verbose=verbose)

        mediator_proteins = gene_ontology.get_genes_from_go_hierarchy(go_annotations,
                                                                      go_terms,
                                                                      mediator_go_terms,
                                                                      go_header=go_header,
                                                                      gene_header=gene_header,
                                                                      verbose=verbose)
    else:
        contact_proteins = gene_ontology.get_genes_from_go_terms(go_annotations,
                                                                 contact_go_terms,
                                                                 go_header=go_header,
                                                                 gene_header=gene_header)

        mediator_proteins = gene_ontology.get_genes_from_go_terms(go_annotations,
                                                                  mediator_go_terms,
                                                                  go_header=go_header,
                                                                  gene_header=gene_header)

    # Avoid same genes in list
    contact_proteins = list(set(contact_proteins) - set(mediator_proteins))

    ppi_dict = get_ppi_dict_from_proteins(ppi_data,
                                          contact_proteins,
                                          mediator_proteins,
                                          interaction_columns=interaction_columns)

    return ppi_dict