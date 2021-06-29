# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import networkx


def get_genes_from_go_terms(go_annotations, go_filter, go_header='GO', gene_header='Gene', verbose=True):
    '''
    Finds genes associated with specific GO-terms.

    Parameters
    ----------
    go_annotations : pandas.DataFrame
        Dataframe containing information about GO term annotations of each
        gene for a given organism according to the ga file. Can be loading
        with the function cell2cell.io.read_data.load_go_annotations().

    go_filter : list
        List containing one or more GO-terms to find associated genes.

    go_header : str, default='GO'
        Column name wherein GO terms are located in the dataframe.

    gene_header : str, default='Gene'
        Column name wherein genes are located in the dataframe.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    genes : list
        List of genes that are associated with GO-terms contained in
        go_filter.
    '''
    if verbose:
        print('Filtering genes by using GO terms')
    genes = list(go_annotations.loc[go_annotations[go_header].isin(go_filter)][gene_header].unique())
    return genes


def get_genes_from_go_hierarchy(go_annotations, go_terms, go_filter, go_header='GO', gene_header='Gene', verbose=False):
    '''
    Obtains genes associated with specific GO terms and their
    children GO terms (below in the hierarchy).

    Parameters
    ----------
    go_annotations : pandas.DataFrame
        Dataframe containing information about GO term annotations of each
        gene for a given organism according to the ga file. Can be loading
        with the function cell2cell.io.read_data.load_go_annotations().

    go_terms : networkx.Graph
        NetworkX Graph containing GO terms datasets from .obo file.
        It could be loaded using
        cell2cell.io.read_data.load_go_terms(filename).

    go_filter : list
        List containing one or more GO-terms to find associated genes.

    go_header : str, default='GO'
        Column name wherein GO terms are located in the dataframe.

    gene_header : str, default='Gene'
        Column name wherein genes are located in the dataframe.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    Returns
    -------
    genes : list
        List of genes that are associated with GO-terms contained in
        go_filter, and related to the children GO terms of those terms.
    '''
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
    '''
    Finds all children GO terms (below in hierarchy) of
    a given GO term.

    Parameters
    ----------
    go_terms : networkx.Graph
        NetworkX Graph containing GO terms datasets from .obo file.
        It could be loaded using
        cell2cell.io.read_data.load_go_terms(filename).

    go_term_name : str
        Specific GO term to find their children. For example:
        'GO:0007155'.

    output_list : list
        List used to perform a Depth First Search and find the
        children in a recursive way. Here the children will be
        automatically written.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.
    '''
    for child in networkx.ancestors(go_terms, go_term_name):
        if child not in output_list:
            if verbose:
                print('Retrieving children for ' + go_term_name)
            output_list.append(child)
        find_all_children_of_go_term(go_terms, child, output_list, verbose)


def find_go_terms_from_keyword(go_terms, keyword, verbose=False):
    '''
    Uses a keyword to find related GO terms.

    Parameters
    ----------
    go_terms : networkx.Graph
        NetworkX Graph containing GO terms datasets from .obo file.
        It could be loaded using
        cell2cell.io.read_data.load_go_terms(filename).

    keyword : str
        Keyword to be included in the names of retrieved GO terms.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    Returns
    -------
    go_filter : list
        List containing all GO terms related to a keyword.
    '''
    go_filter = []
    for go, node in go_terms.nodes.items():
        if keyword in node['name']:
            go_filter.append(go)
            if verbose:
                print(go, node['name'])
    return go_filter