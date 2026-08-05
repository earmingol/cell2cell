# -*- coding: utf-8 -*-

'''Tests for cell2cell.analysis.cell2cell_pipelines and tensor_pipelines'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c


# ---------------------------------------------------------------------------------
# BulkInteractions
# ---------------------------------------------------------------------------------

def test_bulk_interactions_cci_matrix(bulk_interactions, toy_rnaseq):
    cci = bulk_interactions.interaction_space.interaction_elements['cci_matrix']
    assert set(cci.index) == set(toy_rnaseq.columns)
    assert np.allclose(cci.values, cci.values.T)


def test_bulk_interactions_cells_are_naturally_ordered(toy_ppi):
    renamed = c2c.datasets.generate_toy_rnaseq().rename(
        columns={'C3': 'C10', 'C4': 'C20', 'C5': 'C3'})
    interactions = c2c.analysis.BulkInteractions(rnaseq_data=renamed, ppi_data=toy_ppi,
                                                complex_sep=None, verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    cci = interactions.interaction_space.interaction_elements['cci_matrix']
    assert list(cci.columns) == ['C1', 'C2', 'C3', 'C10', 'C20']


def test_bulk_interactions_communication_matrix(bulk_interactions):
    communication = bulk_interactions.interaction_space.interaction_elements['communication_matrix']
    assert communication.shape[1] > 0
    assert np.isfinite(communication.values.astype(float)).all()


@pytest.mark.parametrize('cci_score', ['bray_curtis', 'jaccard', 'count', 'icellnet'])
def test_bulk_interactions_cci_scores(toy_rnaseq, toy_ppi, cci_score):
    interactions = c2c.analysis.BulkInteractions(rnaseq_data=toy_rnaseq, ppi_data=toy_ppi,
                                                cci_score=cci_score, complex_sep=None,
                                                verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    cci = interactions.interaction_space.interaction_elements['cci_matrix']
    assert cci.shape == (toy_rnaseq.shape[1], toy_rnaseq.shape[1])
    distance = interactions.interaction_space.distance_matrix
    assert (distance.values >= 0).all()


@pytest.mark.parametrize('communication_score', ['expression_thresholding',
                                                 'expression_product',
                                                 'expression_mean',
                                                 'expression_gmean'])
def test_bulk_interactions_communication_scores(toy_rnaseq, toy_ppi,
                                                communication_score):
    interactions = c2c.analysis.BulkInteractions(
        rnaseq_data=toy_rnaseq, ppi_data=toy_ppi,
        communication_score=communication_score, complex_sep=None, verbose=False)
    interactions.compute_pairwise_communication_scores(verbose=False)
    matrix = interactions.interaction_space.interaction_elements['communication_matrix']
    assert matrix.shape[1] > 0


@pytest.mark.parametrize('cci_type', ['directed', 'undirected'])
def test_bulk_interactions_cci_types(toy_rnaseq, toy_ppi, cci_type):
    interactions = c2c.analysis.BulkInteractions(rnaseq_data=toy_rnaseq, ppi_data=toy_ppi,
                                                cci_type=cci_type, complex_sep=None,
                                                verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    cci = interactions.interaction_space.interaction_elements['cci_matrix']
    if cci_type == 'undirected':
        assert np.allclose(cci.values, cci.values.T)


def test_bulk_interactions_with_complexes(toy_rnaseq, toy_ppi_complex):
    interactions = c2c.analysis.BulkInteractions(rnaseq_data=toy_rnaseq,
                                                ppi_data=toy_ppi_complex,
                                                complex_sep='&', verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    cci = interactions.interaction_space.interaction_elements['cci_matrix']
    assert cci.shape[0] == toy_rnaseq.shape[1]


def test_bulk_interactions_is_reproducible(toy_rnaseq, toy_ppi, toy_metadata):
    def build():
        interactions = c2c.analysis.BulkInteractions(
            rnaseq_data=toy_rnaseq, ppi_data=toy_ppi, metadata=toy_metadata,
            complex_sep=None, verbose=False)
        interactions.compute_pairwise_cci_scores(verbose=False)
        return interactions.interaction_space.interaction_elements['cci_matrix']

    pd.testing.assert_frame_equal(build(), build())


def test_bulk_interactions_subset_of_cells(toy_rnaseq, toy_ppi):
    '''BulkInteractions has no `excluded_cells`; cells are dropped upstream.'''
    interactions = c2c.analysis.BulkInteractions(
        rnaseq_data=toy_rnaseq.drop(columns=['C1']), ppi_data=toy_ppi,
        complex_sep=None, verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    cci = interactions.interaction_space.interaction_elements['cci_matrix']
    assert 'C1' not in cci.columns
    assert cci.shape[0] == toy_rnaseq.shape[1] - 1


# ---------------------------------------------------------------------------------
# SingleCellInteractions
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize('aggregation_method', ['average', 'nn_cell_fraction', 'trimean'])
def test_single_cell_interactions_aggregation(toy_single_cells, toy_ppi,
                                              aggregation_method):
    rnaseq, metadata = toy_single_cells
    interactions = c2c.analysis.SingleCellInteractions(
        rnaseq_data=rnaseq, ppi_data=toy_ppi, metadata=metadata,
        barcode_col='barcodes', celltype_col='cell_types',
        aggregation_method=aggregation_method, complex_sep=None, verbose=False)
    assert list(interactions.aggregated_expression.columns) == ['CT-1', 'CT-2', 'CT-3']
    assert list(interactions.aggregated_expression.index) == list(rnaseq.index)


def test_single_cell_interactions_computes_scores(toy_single_cells, toy_ppi):
    rnaseq, metadata = toy_single_cells
    interactions = c2c.analysis.SingleCellInteractions(
        rnaseq_data=rnaseq, ppi_data=toy_ppi, metadata=metadata,
        barcode_col='barcodes', celltype_col='cell_types', complex_sep=None,
        verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    cci = interactions.interaction_space.interaction_elements['cci_matrix']
    assert set(cci.index) == {'CT-1', 'CT-2', 'CT-3'}


def test_single_cell_interactions_does_not_modify_the_input(toy_single_cells, toy_ppi):
    rnaseq, metadata = toy_single_cells
    before = rnaseq.copy()
    c2c.analysis.SingleCellInteractions(
        rnaseq_data=rnaseq, ppi_data=toy_ppi, metadata=metadata,
        barcode_col='barcodes', celltype_col='cell_types', complex_sep=None,
        verbose=False)
    pd.testing.assert_frame_equal(rnaseq, before)


@pytest.mark.slow
def test_single_cell_permutation(toy_single_cells, toy_ppi):
    rnaseq, metadata = toy_single_cells
    interactions = c2c.analysis.SingleCellInteractions(
        rnaseq_data=rnaseq, ppi_data=toy_ppi, metadata=metadata,
        barcode_col='barcodes', celltype_col='cell_types', complex_sep=None,
        verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    interactions.permute_cell_labels(evaluation='interactions', permutations=3,
                                     random_state=0, verbose=False)
    pvalues = interactions.cci_permutation_pvalues
    assert ((pvalues.values >= 0) & (pvalues.values <= 1)).all()


# ---------------------------------------------------------------------------------
# initialize_interaction_space
# ---------------------------------------------------------------------------------

def test_initialize_interaction_space_returns_a_space(interaction_space):
    assert hasattr(interaction_space, 'interaction_elements')
    assert hasattr(interaction_space, 'distance_matrix')


# ---------------------------------------------------------------------------------
# run_tensor_cell2cell_pipeline
# ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_run_tensor_cell2cell_pipeline(interaction_tensor):
    metadata = c2c.tensor.generate_tensor_metadata(
        interaction_tensor=interaction_tensor,
        metadata_dicts=[None, None, None, None],
        fill_with_order_elements=True)
    result = c2c.analysis.run_tensor_cell2cell_pipeline(
        interaction_tensor, tensor_metadata=metadata, rank=2,
        tf_optimization='regular', random_state=0, output_folder=None,
        output_fig=False, elbow_metric='error', smooth_elbow=False, upper_rank=3)
    assert result.rank == 2
    assert len(result.factors) == 4


@pytest.mark.slow
def test_run_tensor_cell2cell_pipeline_copies_when_asked(interaction_tensor):
    metadata = c2c.tensor.generate_tensor_metadata(
        interaction_tensor=interaction_tensor,
        metadata_dicts=[None, None, None, None],
        fill_with_order_elements=True)
    result = c2c.analysis.run_tensor_cell2cell_pipeline(
        interaction_tensor, tensor_metadata=metadata, rank=2, copy_tensor=True,
        random_state=0, output_folder=None, output_fig=False)
    assert result is not interaction_tensor
    assert interaction_tensor.rank is None
