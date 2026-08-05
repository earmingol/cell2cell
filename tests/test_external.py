# -*- coding: utf-8 -*-

'''Tests for cell2cell.external'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.external import gseapy as gseapy_module
from cell2cell.external import pcoa as pcoa_module


# ---------------------------------------------------------------------------------
# pcoa
# ---------------------------------------------------------------------------------

def test_pcoa_returns_the_expected_keys(toy_distance):
    result = c2c.external.pcoa(toy_distance)
    for key in ['samples', 'eigvals', 'proportion_explained']:
        assert key in result


def test_pcoa_sample_shape_and_labels(toy_distance):
    result = c2c.external.pcoa(toy_distance)
    assert result['samples'].shape[0] == toy_distance.shape[0]
    assert list(result['samples'].index) == list(toy_distance.index)


def test_pcoa_proportion_explained_is_a_distribution(toy_distance):
    result = c2c.external.pcoa(toy_distance)
    proportions = np.asarray(result['proportion_explained'])
    assert (proportions >= -1e-9).all()
    assert np.isclose(proportions.sum(), 1.0, atol=1e-6)


def test_pcoa_eigenvalues_are_sorted_descending(toy_distance):
    result = c2c.external.pcoa(toy_distance)
    eigvals = np.asarray(result['eigvals'])
    assert np.all(np.diff(eigvals) <= 1e-9)


def test_pcoa_with_a_limited_number_of_dimensions(toy_distance):
    result = c2c.external.pcoa(toy_distance, number_of_dimensions=2)
    assert result['samples'].shape[1] == 2


def test_pcoa_rejects_asymmetric_input(toy_distance):
    asymmetric = toy_distance.copy()
    asymmetric.iloc[0, 1] = 42.0
    with pytest.raises(ValueError):
        c2c.external.pcoa(asymmetric)


def test_pcoa_is_deterministic(toy_distance):
    first = c2c.external.pcoa(toy_distance)
    second = c2c.external.pcoa(toy_distance)
    assert np.allclose(first['samples'].values, second['samples'].values)


def test_pcoa_biplot_runs(toy_distance, toy_rnaseq):
    ordination = c2c.external.pcoa(toy_distance)
    # pcoa_biplot expects the samples frame, not the whole result dictionary
    features = toy_rnaseq.T
    result = c2c.external.pcoa_biplot(ordination, features)
    assert 'features' in result
    assert list(result['features'].index) == list(features.columns)


def test_check_ordination_accepts_a_pcoa_result(toy_distance):
    ordination = c2c.external.pcoa(toy_distance)
    checked = c2c.external._check_ordination(ordination)
    assert checked is not None


# ---------------------------------------------------------------------------------
# umap
# ---------------------------------------------------------------------------------

def test_run_umap_shape(toy_rnaseq):
    result = c2c.external.run_umap(toy_rnaseq, axis=1, n_neighbors=3, random_state=0)
    assert result.shape[0] == toy_rnaseq.shape[1]
    assert result.shape[1] == 2


def test_run_umap_is_reproducible_with_a_seed(toy_rnaseq):
    first = c2c.external.run_umap(toy_rnaseq, axis=1, n_neighbors=3, random_state=7)
    second = c2c.external.run_umap(toy_rnaseq, axis=1, n_neighbors=3, random_state=7)
    assert np.allclose(first.values, second.values)


def test_run_umap_on_the_other_axis(toy_rnaseq):
    result = c2c.external.run_umap(toy_rnaseq, axis=0, n_neighbors=3, random_state=0)
    assert result.shape[0] == toy_rnaseq.shape[0]


# ---------------------------------------------------------------------------------
# gseapy -- offline paths only
# ---------------------------------------------------------------------------------

def test_load_gmt_from_a_local_file(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    assert pathway_per_gene['Protein-A'] == {'TOYDB_PATHWAY_ONE'}
    assert pathway_per_gene['Protein-B'] == {'TOYDB_PATHWAY_ONE', 'TOYDB_PATHWAY_TWO'}
    assert pathway_per_gene['Protein-F'] == {'TOYDB_PATHWAY_THREE'}


def test_load_gmt_covers_every_listed_gene(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    assert set(pathway_per_gene.keys()) == {'Protein-A', 'Protein-B', 'Protein-C',
                                            'Protein-E', 'Protein-F'}


def test_load_gmt_readable_names(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None,
                                              readable_name=True)
    names = set().union(*pathway_per_gene.values())
    # The DB prefix is dropped and underscores become spaces
    assert all('_' not in name for name in names)


def test_generate_lr_geneset_with_an_injected_annotation(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    lr_list = ['Protein-A^Protein-B', 'Protein-B^Protein-C', 'Protein-E^Protein-F']
    geneset = gseapy_module.generate_lr_geneset(lr_list, lr_sep='^',
                                                pathway_per_gene=pathway_per_gene,
                                                min_pathways=0, max_pathways=10000)
    assert isinstance(geneset, dict)
    assert len(geneset) > 0
    for pathway, pairs in geneset.items():
        for pair in pairs:
            assert pair in lr_list


def test_generate_lr_geneset_respects_min_pathways(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    lr_list = ['Protein-A^Protein-B', 'Protein-B^Protein-C']
    geneset = gseapy_module.generate_lr_geneset(lr_list, lr_sep='^',
                                                pathway_per_gene=pathway_per_gene,
                                                min_pathways=100, max_pathways=10000)
    assert geneset == {}


def test_generate_lr_geneset_with_complexes(tiny_gmt):
    pathway_per_gene = gseapy_module.load_gmt(tiny_gmt, backup_url=None)
    lr_list = ['Protein-A&Protein-B^Protein-C']
    geneset = gseapy_module.generate_lr_geneset(lr_list, complex_sep='&', lr_sep='^',
                                                pathway_per_gene=pathway_per_gene,
                                                min_pathways=0, max_pathways=10000)
    assert isinstance(geneset, dict)


# ---------------------------------------------------------------------------------
# goenrich (vendored ontology parser)
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
