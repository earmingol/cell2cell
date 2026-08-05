# -*- coding: utf-8 -*-

'''Tests for cell2cell.external.gseapy -- offline paths only'''

from cell2cell.external import gseapy as gseapy_module


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
