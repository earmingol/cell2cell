# -*- coding: utf-8 -*-

'''Tests for cell2cell.tensor subset, metrics, factor_manipulation and manipulation'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.tensor import factor_manipulation, metrics, subset, tensor_manipulation


# ---------------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------------

def test_subset_tensor_reduces_a_dimension(interaction_tensor):
    cells = list(interaction_tensor.order_names[2])[:2]
    subsetted = subset.subset_tensor(interaction_tensor, {2: cells})
    assert list(subsetted.order_names[2]) == cells
    assert subsetted.tensor.shape[2] == len(cells)
    # Other dimensions are untouched
    assert subsetted.tensor.shape[0] == interaction_tensor.tensor.shape[0]


def test_subset_tensor_keeps_the_values_of_the_kept_elements(interaction_tensor):
    cells = list(interaction_tensor.order_names[2])
    keep = [cells[2], cells[0]]
    subsetted = subset.subset_tensor(interaction_tensor, {2: keep})
    original = np.asarray(interaction_tensor.tensor)
    reduced = np.asarray(subsetted.tensor)
    for new_index, cell in enumerate(subsetted.order_names[2]):
        old_index = cells.index(cell)
        assert np.allclose(reduced[:, :, new_index, :], original[:, :, old_index, :])


def test_subset_tensor_multiple_dimensions(interaction_tensor):
    contexts = list(interaction_tensor.order_names[0])[:2]
    cells = list(interaction_tensor.order_names[2])[:2]
    subsetted = subset.subset_tensor(interaction_tensor, {0: contexts, 2: cells})
    assert subsetted.tensor.shape[0] == 2
    assert subsetted.tensor.shape[2] == 2


def test_subset_tensor_original_order(interaction_tensor):
    cells = list(interaction_tensor.order_names[2])
    reversed_cells = cells[::-1]
    keep_given = subset.subset_tensor(interaction_tensor, {2: reversed_cells},
                                      original_order=False)
    keep_original = subset.subset_tensor(interaction_tensor, {2: reversed_cells},
                                         original_order=True)
    assert list(keep_given.order_names[2]) == reversed_cells
    assert list(keep_original.order_names[2]) == cells


def test_subset_tensor_also_subsets_the_masks(interaction_tensor):
    cells = list(interaction_tensor.order_names[2])[:2]
    subsetted = subset.subset_tensor(interaction_tensor, {2: cells})
    assert np.asarray(subsetted.loc_nans).shape == np.asarray(subsetted.tensor).shape
    assert np.asarray(subsetted.loc_zeros).shape == np.asarray(subsetted.tensor).shape


def test_subset_tensor_does_not_modify_the_original(interaction_tensor):
    before = np.asarray(interaction_tensor.tensor).copy()
    subset.subset_tensor(interaction_tensor, {2: list(interaction_tensor.order_names[2])[:2]})
    assert np.allclose(np.asarray(interaction_tensor.tensor), before)


def test_subset_metadata(factorized_tensor):
    metadata = c2c.tensor.generate_tensor_metadata(
        interaction_tensor=factorized_tensor,
        metadata_dicts=[None, None, None, None],
        fill_with_order_elements=True)
    cells = list(factorized_tensor.order_names[2])[:2]
    subsetted = subset.subset_tensor(factorized_tensor, {2: cells})
    new_metadata = subset.subset_metadata(metadata, subsetted)
    assert list(new_metadata[2]['Element']) == cells


# ---------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------

def test_correlation_index_of_identical_factors_is_zero(factorized_tensor):
    factors = factorized_tensor.factors
    result = metrics.correlation_index(factors, factors)
    assert np.isclose(result, 0.0, atol=1e-6)


def test_correlation_index_is_symmetric(factorized_tensor, interaction_tensor):
    first = factorized_tensor.factors
    interaction_tensor.compute_tensor_factorization(rank=3, random_state=99)
    second = interaction_tensor.factors
    forward = metrics.correlation_index(first, second)
    backward = metrics.correlation_index(second, first)
    assert np.isclose(forward, backward)


def test_correlation_index_is_bounded(factorized_tensor, interaction_tensor):
    interaction_tensor.compute_tensor_factorization(rank=3, random_state=5)
    result = metrics.correlation_index(factorized_tensor.factors,
                                       interaction_tensor.factors)
    assert 0.0 <= result <= 1.0


def test_pairwise_correlation_index(factorized_tensor, interaction_tensor):
    first = factorized_tensor.factors
    interaction_tensor.compute_tensor_factorization(rank=3, random_state=11)
    second = interaction_tensor.factors
    result = metrics.pairwise_correlation_index([first, second])
    assert result.shape == (2, 2)
    assert np.allclose(np.diag(result.values), 0.0, atol=1e-6)
    assert np.allclose(result.values, result.values.T)


# ---------------------------------------------------------------------------------
# factor_manipulation
# ---------------------------------------------------------------------------------

def test_normalize_factors_gives_unit_norm_columns(factorized_tensor):
    normalized = factor_manipulation.normalize_factors(factorized_tensor.factors)
    for frame in normalized.values():
        norms = np.linalg.norm(frame.values, axis=0)
        assert np.allclose(norms, 1.0)


def test_normalize_factors_preserves_labels(factorized_tensor):
    normalized = factor_manipulation.normalize_factors(factorized_tensor.factors)
    assert list(normalized.keys()) == list(factorized_tensor.factors.keys())
    for key, frame in normalized.items():
        assert list(frame.index) == list(factorized_tensor.factors[key].index)
        assert list(frame.columns) == list(factorized_tensor.factors[key].columns)


def test_normalize_factors_keeps_directions(factorized_tensor):
    normalized = factor_manipulation.normalize_factors(factorized_tensor.factors)
    original = factorized_tensor.factors['Sender Cells']['Factor 1'].values
    scaled = normalized['Sender Cells']['Factor 1'].values
    # Same direction, so the correlation must be 1
    assert np.isclose(np.corrcoef(original, scaled)[0, 1], 1.0)


# ---------------------------------------------------------------------------------
# tensor_manipulation
# ---------------------------------------------------------------------------------

def test_concatenate_interaction_tensors(toy_contexts, toy_ppi):
    matrices = list(toy_contexts.values())
    names = list(toy_contexts.keys())
    first = c2c.tensor.InteractionTensor(rnaseq_matrices=matrices[:2], ppi_data=toy_ppi,
                                         context_names=names[:2], how='inner',
                                         complex_sep=None, verbose=False)
    second = c2c.tensor.InteractionTensor(rnaseq_matrices=matrices[2:], ppi_data=toy_ppi,
                                          context_names=names[2:], how='inner',
                                          complex_sep=None, verbose=False)
    labels = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
    combined = tensor_manipulation.concatenate_interaction_tensors(
        [first, second], axis=0, order_labels=labels)
    assert combined.tensor.shape[0] == 4
    assert list(combined.order_names[0]) == names


def test_concatenated_tensor_keeps_the_original_values(toy_contexts, toy_ppi):
    matrices = list(toy_contexts.values())
    names = list(toy_contexts.keys())
    first = c2c.tensor.InteractionTensor(rnaseq_matrices=matrices[:2], ppi_data=toy_ppi,
                                         context_names=names[:2], how='inner',
                                         complex_sep=None, verbose=False)
    second = c2c.tensor.InteractionTensor(rnaseq_matrices=matrices[2:], ppi_data=toy_ppi,
                                          context_names=names[2:], how='inner',
                                          complex_sep=None, verbose=False)
    labels = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
    combined = tensor_manipulation.concatenate_interaction_tensors(
        [first, second], axis=0, order_labels=labels)
    assert np.allclose(np.asarray(combined.tensor)[:2],
                       np.asarray(first.tensor))


def test_concatenated_tensor_can_be_factorized(toy_contexts, toy_ppi):
    matrices = list(toy_contexts.values())
    names = list(toy_contexts.keys())
    tensors = [c2c.tensor.InteractionTensor(rnaseq_matrices=matrices[i:i + 2],
                                            ppi_data=toy_ppi,
                                            context_names=names[i:i + 2], how='inner',
                                            complex_sep=None, verbose=False)
               for i in (0, 2)]
    labels = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
    combined = tensor_manipulation.concatenate_interaction_tensors(tensors, axis=0,
                                                                  order_labels=labels)
    combined.compute_tensor_factorization(rank=2, random_state=0)
    assert len(combined.factors) == 4
