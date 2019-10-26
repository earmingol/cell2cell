# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import networkx

### Gene Ontology related functions
def get_genes_from_go_terms(go_annotations, go_filter, go_header='GO', gene_header='Gene', verbose=True):
    if verbose:
        print('Filtering genes by using GO terms')
    genes = list(np.unique(go_annotations.loc[go_annotations[go_header].isin(go_filter)][gene_header].values))
    return genes


def get_genes_from_go_hierarchy(go_annotations, go_terms, go_filter, go_header='GO', gene_header='Gene', verbose=False):
    go_hierarchy = go_filter.copy()
    iter = len(go_hierarchy)
    for i in range(iter):
        find_all_children_of_go_term(go_terms, go_hierarchy[i], go_hierarchy, verbose=verbose)
    go_hierarchy = list(set(go_hierarchy))
    genes = get_genes_from_go_terms(go_annotations=go_annotations,
                                    go_filter=go_hierarchy,
                                    go_header=go_header,
                                    gene_header=gene_header,
                                    verbose=verbose)
    return genes


def find_all_children_of_go_term(go_terms, go_term_name, output_list, verbose=True):
    for child in networkx.ancestors(go_terms, go_term_name):
        if child not in output_list:
            if verbose:
                print('Retrieving children for ' + go_term_name)
            output_list.append(child)
        find_all_children_of_go_term(go_terms, child, output_list, verbose)


def find_go_terms_from_keyword(go_terms, keyword, verbose=False):
    '''
    This function looks for GO terms containing a given keyword.

    Parameters
    ----------
    go_terms : networkx.Graph()
        NetworkX Graph containing GO terms datasets from .obo file. It could be loaded using the goenrich package or the
        function cell2cell.io.load_go_terms(filename).

    keyword : str
        Keyword to be included in the names of retrieved GO terms.

    Returns
    -------
    go_filter : array-like
        List containing all GO terms that have keyword as part of their names.
    '''
    go_filter = []
    for go, node in go_terms.nodes.items():
        if keyword in node['name']:
            go_filter.append(go)
            if verbose:
                print(go, node['name'])
    return go_filter