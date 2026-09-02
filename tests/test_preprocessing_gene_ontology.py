# -*- coding: utf-8 -*-

'''Tests for cell2cell.preprocessing.gene_ontology'''

import networkx as nx
import pandas as pd
import pytest

from cell2cell.preprocessing import gene_ontology


HEADERS = dict(go_header='go_id', gene_header='db_object_symbol')


def test_get_genes_from_go_terms_single_term(go_annotations):
    genes = gene_ontology.get_genes_from_go_terms(go_annotations,
                                                  go_filter=['GO:0000003'],
                                                  verbose=False, **HEADERS)
    assert set(genes) == {'Protein-B', 'Protein-E'}


def test_get_genes_from_go_terms_multiple_terms(go_annotations):
    genes = gene_ontology.get_genes_from_go_terms(
        go_annotations, go_filter=['GO:0000002', 'GO:0000004'], verbose=False, **HEADERS)
    assert set(genes) == {'Protein-A', 'Protein-C', 'Protein-F'}


def test_get_genes_from_go_terms_unknown_term(go_annotations):
    genes = gene_ontology.get_genes_from_go_terms(go_annotations,
                                                  go_filter=['GO:9999999'],
                                                  verbose=False, **HEADERS)
    assert genes == []


def test_find_all_children_of_go_term(go_terms_graph):
    children = []
    gene_ontology.find_all_children_of_go_term(go_terms_graph, 'GO:0000001', children,
                                               verbose=False)
    # GO:0000002 and GO:0000003 are children, GO:0000004 is a grandchild
    assert set(children) == {'GO:0000002', 'GO:0000003', 'GO:0000004'}


def test_find_all_children_of_a_leaf(go_terms_graph):
    children = []
    gene_ontology.find_all_children_of_go_term(go_terms_graph, 'GO:0000004', children,
                                               verbose=False)
    assert children == []


def test_get_genes_from_go_hierarchy_includes_descendants(go_annotations,
                                                          go_terms_graph):
    genes = gene_ontology.get_genes_from_go_hierarchy(
        go_annotations=go_annotations, go_terms=go_terms_graph,
        go_filter=['GO:0000001'], verbose=False, **HEADERS)
    # The root plus all of its descendants covers every annotated gene
    assert set(genes) == {'Protein-A', 'Protein-B', 'Protein-C', 'Protein-D',
                          'Protein-E', 'Protein-F'}


def test_get_genes_from_go_hierarchy_of_a_subtree(go_annotations, go_terms_graph):
    genes = gene_ontology.get_genes_from_go_hierarchy(
        go_annotations=go_annotations, go_terms=go_terms_graph,
        go_filter=['GO:0000002'], verbose=False, **HEADERS)
    # GO:0000002 annotates Protein-A, its child GO:0000004 annotates C and F
    assert set(genes) == {'Protein-A', 'Protein-C', 'Protein-F'}


def test_get_genes_from_go_hierarchy_is_reproducible(go_annotations, go_terms_graph):
    kwargs = dict(go_annotations=go_annotations, go_terms=go_terms_graph,
                  go_filter=['GO:0000001'], verbose=False, **HEADERS)
    assert (gene_ontology.get_genes_from_go_hierarchy(**kwargs) ==
            gene_ontology.get_genes_from_go_hierarchy(**kwargs))


def test_find_go_terms_from_keyword(go_terms_graph):
    terms = gene_ontology.find_go_terms_from_keyword(go_terms_graph, 'adhesion',
                                                     verbose=False)
    assert terms == ['GO:0000002']


def test_find_go_terms_from_keyword_without_matches(go_terms_graph):
    assert gene_ontology.find_go_terms_from_keyword(go_terms_graph, 'nonsense',
                                                    verbose=False) == []


def test_find_go_terms_from_keyword_matches_several(go_terms_graph):
    terms = gene_ontology.find_go_terms_from_keyword(go_terms_graph, 'toy',
                                                     verbose=False)
    assert len(terms) == 4
