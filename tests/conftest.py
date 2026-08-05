# -*- coding: utf-8 -*-

'''Shared fixtures for the cell2cell test suite.

Fixtures are built on the toy datasets in `cell2cell.datasets` whenever possible, so
the tests are deterministic and do not need any download. Test-only scaffolding that
is not useful enough to be part of the public API (a small gene-ontology graph, a tiny
GMT file) is defined here instead.
'''

import matplotlib

# Must be set before anything imports pyplot, otherwise the tests try to open windows.
matplotlib.use('Agg')

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import cell2cell as c2c


# ---------------------------------------------------------------------------------
# Global hygiene
# ---------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_figures():
    '''Closes every figure after each test, so plotting tests do not leak state.'''
    yield
    plt.close('all')


# ---------------------------------------------------------------------------------
# Existing toy datasets
# ---------------------------------------------------------------------------------

@pytest.fixture
def toy_rnaseq():
    '''Toy bulk RNA-seq dataset. Genes (Protein-A..F) x cells (C1..C5).'''
    return c2c.datasets.generate_toy_rnaseq()


@pytest.fixture
def toy_ppi():
    '''Toy list of protein-protein interactions. Columns A, B, score.'''
    return c2c.datasets.generate_toy_ppi()


@pytest.fixture
def toy_ppi_complex():
    '''Toy list of PPIs including multimeric complexes, separated by "&".'''
    return c2c.datasets.generate_toy_ppi(prot_complex=True)


@pytest.fixture
def toy_metadata():
    '''Toy metadata for cells C1..C5. Columns #SampleID, Groups.'''
    return c2c.datasets.generate_toy_metadata()


@pytest.fixture
def toy_distance():
    '''Toy square cell-cell distance matrix for cells C1..C5.'''
    return c2c.datasets.generate_toy_distance()


# ---------------------------------------------------------------------------------
# Toy datasets added in v0.9.0
# ---------------------------------------------------------------------------------

@pytest.fixture
def toy_contexts():
    '''Dictionary with a toy RNA-seq dataset for each of 4 contexts.'''
    return c2c.datasets.generate_toy_contexts()


@pytest.fixture
def toy_single_cells():
    '''Tuple (rnaseq, metadata) for a toy single-cell dataset. 3 cell types.'''
    return c2c.datasets.generate_toy_single_cells()


@pytest.fixture
def toy_coordinates():
    '''Toy spatial coordinates. Columns X, Y, celltype.'''
    return c2c.datasets.generate_toy_coordinates()


@pytest.fixture
def toy_spatial_adata():
    '''Toy AnnData with coordinates in obsm["spatial"] on a 15x15 lattice.'''
    return c2c.datasets.generate_toy_spatial_adata()


@pytest.fixture
def toy_liana():
    '''Toy LIANA-like output in long format. 3 contexts x 3 cell types.'''
    return c2c.datasets.generate_toy_liana_output()


# ---------------------------------------------------------------------------------
# Derived objects used across several test modules
# ---------------------------------------------------------------------------------

@pytest.fixture
def bulk_interactions(toy_rnaseq, toy_ppi, toy_metadata):
    '''A BulkInteractions object with CCI and communication scores computed.'''
    interactions = c2c.analysis.BulkInteractions(rnaseq_data=toy_rnaseq,
                                                 ppi_data=toy_ppi,
                                                 metadata=toy_metadata,
                                                 interaction_columns=('A', 'B'),
                                                 communication_score='expression_product',
                                                 cci_score='bray_curtis',
                                                 cci_type='undirected',
                                                 complex_sep=None,
                                                 verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    interactions.compute_pairwise_communication_scores(verbose=False)
    return interactions


@pytest.fixture
def analysis_setup():
    '''Analysis parameters expected by `initialize_interaction_space`.'''
    return {'communication_score': 'expression_product',
            'cci_score': 'bray_curtis',
            'cci_type': 'undirected',
            'ccc_type': 'undirected'}


@pytest.fixture
def cutoff_setup():
    '''Cutoff parameters expected by `initialize_interaction_space`.'''
    return {'type': 'constant_value', 'parameter': 10}


@pytest.fixture
def interaction_space(toy_rnaseq, toy_ppi, analysis_setup, cutoff_setup):
    '''An InteractionSpace with the CCI and communication matrices computed.'''
    space = c2c.analysis.initialize_interaction_space(rnaseq_data=toy_rnaseq,
                                                     ppi_data=toy_ppi,
                                                     cutoff_setup=cutoff_setup,
                                                     analysis_setup=analysis_setup,
                                                     complex_sep=None,
                                                     verbose=False)
    space.compute_pairwise_cci_scores(verbose=False)
    space.compute_pairwise_communication_scores(verbose=False)
    return space


@pytest.fixture
def toy_cells(toy_rnaseq):
    '''Dictionary of Cell objects, one per cell in the toy RNA-seq dataset.'''
    return c2c.core.get_cells_from_rnaseq(toy_rnaseq, verbose=False)


@pytest.fixture
def interaction_tensor(toy_contexts, toy_ppi):
    '''A 4D InteractionTensor built from the toy contexts. Not yet factorized.'''
    return c2c.tensor.InteractionTensor(rnaseq_matrices=list(toy_contexts.values()),
                                        ppi_data=toy_ppi,
                                        context_names=list(toy_contexts.keys()),
                                        how='inner',
                                        complex_sep=None,
                                        communication_score='expression_product',
                                        verbose=False)


@pytest.fixture
def factorized_tensor(interaction_tensor):
    '''An InteractionTensor already decomposed with rank=3 and a fixed seed.'''
    interaction_tensor.compute_tensor_factorization(rank=3, random_state=0)
    return interaction_tensor


@pytest.fixture
def prebuilt_tensor():
    '''A small deterministic PreBuiltTensor with 4 dimensions.

    Sender and receiver cells are deliberately NOT in alphabetical order, so the
    tests can detect functions that assume a sorted tensor.
    '''
    cells = ['C3', 'C1', 'C2']
    lr_pairs = ['Protein-A^Protein-B', 'Protein-B^Protein-C', 'Protein-C^Protein-A']
    contexts = ['Context-1', 'Context-2']
    shape = (len(contexts), len(lr_pairs), len(cells), len(cells))
    # Deterministic, strictly positive values
    tensor = (np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.) / np.prod(shape)
    return c2c.tensor.PreBuiltTensor(tensor=tensor,
                                     order_names=[contexts, lr_pairs, cells, cells],
                                     order_labels=['Contexts', 'Ligand-Receptor Pairs',
                                                   'Sender Cells', 'Receiver Cells'])


@pytest.fixture
def factorized_prebuilt_tensor(prebuilt_tensor):
    '''The unsorted PreBuiltTensor, decomposed with rank=2 and a fixed seed.'''
    prebuilt_tensor.compute_tensor_factorization(rank=2, random_state=0)
    return prebuilt_tensor


# ---------------------------------------------------------------------------------
# Test-only scaffolding (not part of the public API)
# ---------------------------------------------------------------------------------

@pytest.fixture
def go_terms_graph():
    '''A small gene-ontology hierarchy as a networkx.DiGraph.

    Edges point from the children to their parents, as in the ontologies parsed by
    `cell2cell.external.goenrich.ontology`. 'GO:0000001' is the root, with two
    children and one grandchild.
    '''
    graph = nx.DiGraph()
    edges = [('GO:0000002', 'GO:0000001'),
             ('GO:0000003', 'GO:0000001'),
             ('GO:0000004', 'GO:0000002')]
    graph.add_edges_from(edges)
    names = {'GO:0000001': 'toy root term',
             'GO:0000002': 'toy cell adhesion',
             'GO:0000003': 'toy extracellular space',
             'GO:0000004': 'toy cell junction'}
    for node, name in names.items():
        graph.add_node(node, name=name)
    return graph


@pytest.fixture
def go_annotations():
    '''GO annotations for the toy genes, as returned by `goenrich.goa`.'''
    records = [('Protein-A', 'GO:0000002'),
               ('Protein-B', 'GO:0000003'),
               ('Protein-C', 'GO:0000004'),
               ('Protein-D', 'GO:0000001'),
               ('Protein-E', 'GO:0000003'),
               ('Protein-F', 'GO:0000004')]
    return pd.DataFrame(records, columns=['db_object_symbol', 'go_id'])


@pytest.fixture
def tiny_gmt(tmp_path):
    '''Path to a minimal GMT file, to test the offline path of `load_gmt`.

    The second field of a GMT line is a description. `load_gmt` discards it only when
    it looks like a URL, which is the case for the MSigDB files it is written for, so
    the fixture uses URLs to mirror those files.
    '''
    path = tmp_path / 'toy-pathways.gmt'
    lines = ['TOYDB_PATHWAY_ONE\thttp://toydb.org/one\tProtein-A\tProtein-B\tProtein-C',
             'TOYDB_PATHWAY_TWO\thttp://toydb.org/two\tProtein-B\tProtein-E',
             'TOYDB_PATHWAY_THREE\thttp://toydb.org/three\tProtein-F']
    path.write_text('\n'.join(lines) + '\n')
    return str(path)
