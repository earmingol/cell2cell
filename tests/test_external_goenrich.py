# -*- coding: utf-8 -*-

'''Tests for cell2cell.external.goenrich (vendored ontology and annotation parsers)'''

import pytest

import cell2cell as c2c
from cell2cell.external import goenrich


# ---------------------------------------------------------------------------------
# ontology
# ---------------------------------------------------------------------------------

def test_ontology_parses_a_minimal_obo(tmp_path):
    obo = tmp_path / 'toy.obo'
    obo.write_text('\n'.join([
        'format-version: 1.2',
        '',
        '[Term]',
        'id: GO:0000001',
        'name: toy root',
        'namespace: biological_process',
        '',
        '[Term]',
        'id: GO:0000002',
        'name: toy child',
        'namespace: biological_process',
        'is_a: GO:0000001',
        '',
        '']))
    graph = c2c.external.ontology(str(obo))
    assert 'GO:0000001' in graph.nodes()
    assert 'GO:0000002' in graph.nodes()
    assert graph.has_edge('GO:0000001', 'GO:0000002') or \
           graph.has_edge('GO:0000002', 'GO:0000001')


def test_ontology_keeps_both_is_a_and_part_of_parents(tiny_obo):
    graph = c2c.external.ontology(tiny_obo)
    # `ontology` returns the graph reversed, so edges point from a child to its parent.
    # GO:0000002 is a child through `is_a`, GO:0000003 through `relationship: part_of`.
    assert graph.has_edge('GO:0000002', 'GO:0000001')
    assert graph.has_edge('GO:0000003', 'GO:0000001')


def test_ontology_drops_obsolete_terms(tiny_obo):
    graph = c2c.external.ontology(tiny_obo)
    assert 'GO:0000004' not in graph.nodes()


def test_ontology_accepts_an_open_file_handle(tiny_obo):
    '''`ontology` only closes the file when it opened it itself.'''
    with open(tiny_obo) as handle:
        graph = c2c.external.ontology(handle)
        assert not handle.closed
    assert 'GO:0000002' in graph.nodes()


def test_ontology_annotates_the_depth_of_each_term(tiny_obo):
    graph = c2c.external.ontology(tiny_obo)
    assert graph.nodes['GO:0000001']['depth'] == 0
    assert graph.nodes['GO:0000002']['depth'] == 1


# ---------------------------------------------------------------------------------
# goa / sgd
# ---------------------------------------------------------------------------------

def test_goa_names_the_gaf_columns(tiny_gaf):
    annotations = goenrich.goa(tiny_gaf, experimental=False)
    assert list(annotations.columns) == list(goenrich.GENE_ASSOCIATION_COLUMNS)
    assert len(annotations) == 3


def test_goa_keeps_only_experimental_evidence(tiny_gaf):
    annotations = goenrich.goa(tiny_gaf, experimental=True)
    # Protein-C is annotated with IEA, which is not experimental evidence.
    assert set(annotations['db_object_symbol']) == {'Protein-A', 'Protein-B'}
    assert set(annotations['evidence_code']).issubset(set(goenrich.EXPERIMENTAL_EVIDENCE))


def test_goa_adds_the_evidence_column_to_usecols(tiny_gaf):
    '''`evidence_code` is needed for the filter even when it was not requested.'''
    annotations = goenrich.goa(tiny_gaf, experimental=True,
                               usecols=('db_object_symbol', 'go_id'))
    assert 'evidence_code' in annotations.columns
    assert set(annotations['db_object_symbol']) == {'Protein-A', 'Protein-B'}


def test_sgd_delegates_to_goa(tiny_gaf):
    '''`sgd` is `goa` with `experimental` defaulting to False.'''
    assert goenrich.sgd(tiny_gaf).equals(goenrich.goa(tiny_gaf, experimental=False))


# ---------------------------------------------------------------------------------
# gene2go
# ---------------------------------------------------------------------------------

def test_gene2go_filters_by_taxon(tiny_gene2go):
    annotations = goenrich.gene2go(tiny_gene2go, tax_id=9606)
    assert set(annotations['tax_id']) == {9606}
    assert len(annotations) == 2


def test_gene2go_filters_by_experimental_evidence(tiny_gene2go):
    annotations = goenrich.gene2go(tiny_gene2go, experimental=True, tax_id=9606)
    assert set(annotations['Evidence']) == {'IDA'}
    assert len(annotations) == 1


def test_gene2go_names_its_columns(tiny_gene2go):
    annotations = goenrich.gene2go(tiny_gene2go)
    assert list(annotations.columns) == list(goenrich.GENE2GO_COLUMNS)
