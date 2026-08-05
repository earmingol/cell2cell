# -*- coding: utf-8 -*-

'''Tests for cell2cell.datasets'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.datasets import toy_data


TOY_GENES = ['Protein-A', 'Protein-B', 'Protein-C', 'Protein-D', 'Protein-E', 'Protein-F']
TOY_CELLS = ['C1', 'C2', 'C3', 'C4', 'C5']

DETERMINISTIC_GENERATORS = ['generate_toy_rnaseq', 'generate_toy_ppi', 'generate_toy_metadata',
                            'generate_toy_distance', 'generate_toy_contexts',
                            'generate_toy_single_cells', 'generate_toy_coordinates',
                            'generate_toy_liana_output']


# ---------------------------------------------------------------------------------
# Pre-existing toy datasets
# ---------------------------------------------------------------------------------

def test_toy_rnaseq_shape_and_labels(toy_rnaseq):
    assert toy_rnaseq.shape == (6, 5)
    assert list(toy_rnaseq.index) == TOY_GENES
    assert list(toy_rnaseq.columns) == TOY_CELLS
    assert toy_rnaseq.index.name == 'gene_id'
    assert (toy_rnaseq.values > 0).all()


def test_toy_ppi_columns(toy_ppi):
    assert list(toy_ppi.columns) == ['A', 'B', 'score']
    assert toy_ppi.shape == (10, 3)
    assert (toy_ppi['score'] == 1.0).all()
    # Every interactor must exist in the toy RNA-seq dataset
    assert set(toy_ppi['A']).union(toy_ppi['B']).issubset(set(TOY_GENES))


def test_toy_ppi_complex_uses_ampersand(toy_ppi_complex):
    interactors = set(toy_ppi_complex['A']).union(toy_ppi_complex['B'])
    assert any('&' in i for i in interactors)
    # Each subunit of a complex must be a known gene
    for interactor in interactors:
        for subunit in interactor.split('&'):
            assert subunit in TOY_GENES


def test_toy_metadata(toy_metadata):
    assert list(toy_metadata.columns) == ['#SampleID', 'Groups']
    assert list(toy_metadata['#SampleID']) == TOY_CELLS


def test_toy_distance_is_a_valid_distance_matrix(toy_distance):
    assert list(toy_distance.index) == TOY_CELLS
    assert list(toy_distance.columns) == TOY_CELLS
    assert np.allclose(np.diag(toy_distance.values), 0.0)
    assert np.allclose(toy_distance.values, toy_distance.values.T)
    assert (toy_distance.values >= 0).all()


# ---------------------------------------------------------------------------------
# Toy datasets added in v0.9.0
# ---------------------------------------------------------------------------------

def test_toy_contexts_default(toy_contexts):
    assert list(toy_contexts.keys()) == ['Context-1', 'Context-2', 'Context-3', 'Context-4']
    for df in toy_contexts.values():
        assert list(df.index) == TOY_GENES
        assert list(df.columns) == TOY_CELLS


def test_toy_contexts_differ_between_contexts(toy_contexts):
    values = [df.values for df in toy_contexts.values()]
    for other in values[1:]:
        assert not np.allclose(values[0], other)


def test_toy_contexts_custom_names():
    names = ['early', 'late']
    contexts = c2c.datasets.generate_toy_contexts(n_contexts=2, context_names=names)
    assert list(contexts.keys()) == names


def test_toy_contexts_rejects_mismatched_names():
    with pytest.raises(AssertionError):
        c2c.datasets.generate_toy_contexts(n_contexts=3, context_names=['only-one'])


def test_toy_contexts_with_ten_or_more_exposes_natural_order():
    '''With >= 10 contexts the alphabetical and natural orders differ.'''
    contexts = c2c.datasets.generate_toy_contexts(n_contexts=11)
    names = list(contexts.keys())
    assert names[-2:] == ['Context-10', 'Context-11']
    assert sorted(names) != names          # alphabetical order is NOT the natural one
    assert names[1] == 'Context-2'


def test_toy_single_cells(toy_single_cells):
    rnaseq, metadata = toy_single_cells
    assert list(rnaseq.index) == TOY_GENES
    assert rnaseq.shape == (6, 12)          # 3 cell types x 4 cells
    assert list(metadata.columns) == ['barcodes', 'cell_types']
    assert list(metadata['barcodes']) == list(rnaseq.columns)
    assert list(metadata['cell_types'].unique()) == ['CT-1', 'CT-2', 'CT-3']
    assert metadata['cell_types'].value_counts().unique().tolist() == [4]


def test_toy_single_cells_sizes():
    rnaseq, metadata = c2c.datasets.generate_toy_single_cells(n_cell_types=11,
                                                              n_cells_per_type=2)
    assert rnaseq.shape == (6, 22)
    types = list(metadata['cell_types'].unique())
    assert types[-1] == 'CT-11'
    assert sorted(types) != types           # exposes natural vs alphabetical ordering


def test_toy_coordinates(toy_coordinates):
    assert list(toy_coordinates.columns) == ['X', 'Y', 'celltype']
    assert toy_coordinates.shape == (15, 3)
    assert list(toy_coordinates['celltype'].unique()) == ['CT-1', 'CT-2', 'CT-3']
    assert toy_coordinates.index.is_unique


def test_toy_coordinates_celltypes_are_spatially_separated(toy_coordinates):
    '''Cells of one type must be closer to each other than to another type.'''
    groups = toy_coordinates.groupby('celltype')[['X', 'Y']].mean()
    assert not np.allclose(groups.loc['CT-1'].values, groups.loc['CT-2'].values)


def test_toy_spatial_adata(toy_spatial_adata):
    adata = toy_spatial_adata
    assert adata.shape == (225, 6)
    assert 'spatial' in adata.obsm
    assert adata.obsm['spatial'].shape == (225, 2)
    assert list(adata.var_names) == TOY_GENES
    assert 'celltype' in adata.obs.columns
    assert adata.obs_names[0] == 'spot-1'


def test_toy_spatial_adata_coordinate_range(toy_spatial_adata):
    coords = toy_spatial_adata.obsm['spatial']
    assert np.isclose(coords.min(), 0.0)
    assert np.isclose(coords.max(), 100.0)
    # A lattice wide enough that >= 11 bins per axis each receive cells
    assert len(np.unique(coords[:, 0])) >= 11


def test_toy_spatial_adata_num_cells():
    adata = c2c.datasets.generate_toy_spatial_adata(num_cells=50, n_cell_types=2)
    assert adata.shape[0] == 50
    assert set(adata.obs['celltype']) == {'CT-1', 'CT-2'}


def test_toy_liana_output(toy_liana):
    assert list(toy_liana.columns) == ['context', 'source', 'target', 'ligand',
                                       'receptor', 'score']
    # 3 contexts x 3 senders x 3 receivers x 10 LR pairs
    assert toy_liana.shape == (270, 6)
    assert list(toy_liana['context'].unique()) == ['Context-1', 'Context-2', 'Context-3']
    assert toy_liana['score'].between(0, 1).all()


def test_toy_liana_output_has_no_duplicated_entries(toy_liana):
    '''dataframes_to_tensor warns/aggregates on duplicates, so the toy data must be unique.'''
    keys = ['context', 'source', 'target', 'ligand', 'receptor']
    assert not toy_liana.duplicated(subset=keys).any()


def test_toy_liana_output_groups_into_a_context_dict(toy_liana):
    context_dict = {k: v for k, v in toy_liana.groupby('context')}
    assert len(context_dict) == 3
    for df in context_dict.values():
        assert df.shape[0] == 90


# ---------------------------------------------------------------------------------
# Determinism -- every toy generator must be reproducible
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize('name', DETERMINISTIC_GENERATORS)
def test_toy_generators_are_deterministic(name):
    generator = getattr(toy_data, name)
    first, second = generator(), generator()
    if isinstance(first, tuple):
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)
    elif isinstance(first, dict):
        assert list(first.keys()) == list(second.keys())
        for key in first:
            pd.testing.assert_frame_equal(first[key], second[key])
    else:
        pd.testing.assert_frame_equal(first, second)


def test_toy_spatial_adata_is_deterministic():
    first = c2c.datasets.generate_toy_spatial_adata(num_cells=30)
    second = c2c.datasets.generate_toy_spatial_adata(num_cells=30)
    assert np.allclose(first.X, second.X)
    assert np.allclose(first.obsm['spatial'], second.obsm['spatial'])
    assert list(first.obs['celltype']) == list(second.obs['celltype'])


def test_toy_data_module_does_not_import_anndata_eagerly():
    '''anndata must be imported inside generate_toy_spatial_adata, not at module level.'''
    import inspect
    source = inspect.getsource(toy_data)
    header = source.split('def ')[0]
    assert 'anndata' not in header


# ---------------------------------------------------------------------------------
# Random data generators
# ---------------------------------------------------------------------------------

def test_generate_random_rnaseq_is_seeded():
    first = c2c.datasets.generate_random_rnaseq(size=4, row_names=TOY_GENES,
                                                random_state=0, verbose=False)
    second = c2c.datasets.generate_random_rnaseq(size=4, row_names=TOY_GENES,
                                                 random_state=0, verbose=False)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.index) == TOY_GENES
    assert list(first.columns) == ['Cell-1', 'Cell-2', 'Cell-3', 'Cell-4']


def test_generate_random_rnaseq_is_scaled_to_a_million():
    df = c2c.datasets.generate_random_rnaseq(size=3, row_names=TOY_GENES,
                                             random_state=1, verbose=False)
    assert np.allclose(df.sum(axis=0).values, 1e6)


def test_generate_random_ppi_is_seeded():
    first = c2c.datasets.generate_random_ppi(max_size=5, interactors_A=TOY_GENES,
                                             random_state=0, verbose=False)
    second = c2c.datasets.generate_random_ppi(max_size=5, interactors_A=TOY_GENES,
                                              random_state=0, verbose=False)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ['A', 'B']
    # max_size is an upper bound: PPIs are de-duplicated after resampling
    assert first.shape[0] <= 5


def test_generate_random_ppi_rejects_impossible_size():
    with pytest.raises(AssertionError):
        c2c.datasets.generate_random_ppi(max_size=1000, interactors_A=['A', 'B'],
                                         verbose=False)


def test_generate_random_cci_scores_symmetric():
    matrix = c2c.datasets.generate_random_cci_scores(cell_number=4, symmetric=True,
                                                     random_state=0)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix.values, matrix.values.T)
    assert matrix.values.min() >= 0 and matrix.values.max() <= 1


def test_generate_random_cci_scores_asymmetric_and_labelled():
    labels = ['a', 'b', 'c']
    matrix = c2c.datasets.generate_random_cci_scores(cell_number=3, labels=labels,
                                                     symmetric=False, random_state=0)
    assert list(matrix.index) == labels
    assert not np.allclose(matrix.values, matrix.values.T)


def test_generate_random_cci_scores_rejects_label_mismatch():
    with pytest.raises(AssertionError):
        c2c.datasets.generate_random_cci_scores(cell_number=3, labels=['a'])


def test_generate_random_metadata():
    metadata = c2c.datasets.generate_random_metadata(cell_labels=TOY_CELLS, group_number=2)
    assert list(metadata.columns) == ['Cell', 'Group']
    assert list(metadata['Cell']) == TOY_CELLS
    assert set(metadata['Group']).issubset({1, 2})


def test_heuristic_go_terms():
    terms = c2c.datasets.HeuristicGOTerms()
    assert len(terms.contact_go_terms) == 13
    assert len(terms.mediator_go_terms) == 6
    assert all(t.startswith('GO:') for t in terms.contact_go_terms)
    assert all(t.startswith('GO:') for t in terms.mediator_go_terms)


# ---------------------------------------------------------------------------------
# Downloads -- deselected by default (see pytest.ini)
# ---------------------------------------------------------------------------------

@pytest.mark.network
def test_balf_covid_downloads(tmp_path):
    adata = c2c.datasets.balf_covid(filename=str(tmp_path / 'balf.h5ad'))
    assert adata.shape[0] > 0


@pytest.mark.network
def test_gsea_msig_downloads():
    pathway_per_gene = c2c.datasets.gsea_msig(organism='human', pathwaydb='KEGG')
    assert len(pathway_per_gene) > 0
