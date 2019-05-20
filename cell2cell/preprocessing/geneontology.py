# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import networkx

### Gene Ontology related functions
def get_genes_from_go_terms(go_annotations, go_filter, go_header='GO', gene_header='Gene'):
    print('Filtering genes by using GO terms')
    filtered_genes = list(np.unique(go_annotations.loc[go_annotations[go_header].isin(go_filter)][gene_header].values))
    return filtered_genes


def get_genes_from_parent_go_terms(go_annotations, go_terms, go_filter, go_header='GO', gene_header='Gene', verbose=False):
    parent_GOs = go_filter.copy()
    iter = len(parent_GOs)
    for i in range(iter):
        find_all_children_of_go_term(go_terms, parent_GOs[i], parent_GOs, verbose=verbose)
    genes = get_genes_from_go_terms(go_annotations, parent_GOs, go_header, gene_header)
    return genes


def find_all_children_of_go_term(graph, go_term, empty_list, verbose=True):
    for child in networkx.ancestors(graph, go_term):
        if child not in empty_list:
            if verbose:
                print('Retrieving children for ' + go_term)
            empty_list.append(child)
        find_all_children_of_go_term(graph, child, empty_list, verbose)