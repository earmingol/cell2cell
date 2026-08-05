# -*- coding: utf-8 -*-

'''Tests for cell2cell.tensor.coupled_tensor and coupled_factorization'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.tensor import coupled_factorization


SHARED_CONTEXTS = {'shared': [(0, 0)]}


@pytest.fixture
def two_tensors(toy_contexts, toy_ppi):
    '''Two tensors that share their context dimension.'''
    matrices = list(toy_contexts.values())
    names = list(toy_contexts.keys())

    def build(scale):
        return c2c.tensor.InteractionTensor(
            rnaseq_matrices=[matrix * scale for matrix in matrices],
            ppi_data=toy_ppi, context_names=names, how='inner',
            complex_sep=None, communication_score='expression_product',
            verbose=False)

    return build(1.0), build(1.7)


@pytest.fixture
def coupled(two_tensors):
    first, second = two_tensors
    return c2c.tensor.CoupledInteractionTensor(tensor1=first, tensor2=second,
                                               mode_mapping=SHARED_CONTEXTS)


@pytest.fixture
def factorized_coupled(coupled):
    coupled.compute_tensor_factorization(rank=2, random_state=0)
    return coupled


# ---------------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------------

def test_coupled_tensor_keeps_both_tensors(coupled, two_tensors):
    first, second = two_tensors
    assert coupled.shape is not None
    assert np.asarray(coupled.tensor1).shape == np.asarray(first.tensor).shape
    assert np.asarray(coupled.tensor2).shape == np.asarray(second.tensor).shape


def test_coupled_tensor_shared_mode_elements_agree(coupled):
    assert list(coupled.order_names1[0]) == list(coupled.order_names2[0])


def test_coupled_tensor_rejects_an_empty_mode_mapping(two_tensors):
    first, second = two_tensors
    with pytest.raises(ValueError):
        c2c.tensor.CoupledInteractionTensor(tensor1=first, tensor2=second,
                                            mode_mapping={'shared': []})


def test_coupled_tensor_rejects_out_of_range_modes(two_tensors):
    '''A mode outside the tensor's dimensions is rejected, though the error type is
    an IndexError rather than the validated ValueError.'''
    first, second = two_tensors
    with pytest.raises((ValueError, IndexError)):
        c2c.tensor.CoupledInteractionTensor(tensor1=first, tensor2=second,
                                            mode_mapping={'shared': [(0, 99)]})


def test_coupled_tensor_accepts_several_shared_modes(two_tensors):
    first, second = two_tensors
    instance = c2c.tensor.CoupledInteractionTensor(
        tensor1=first, tensor2=second, mode_mapping={'shared': [(0, 0), (2, 2)]})
    assert instance is not None


def test_coupled_tensor_reorders_a_misaligned_shared_mode(toy_contexts, toy_ppi):
    matrices = list(toy_contexts.values())
    names = list(toy_contexts.keys())
    first = c2c.tensor.InteractionTensor(rnaseq_matrices=matrices, ppi_data=toy_ppi,
                                        context_names=names, how='inner',
                                        complex_sep=None, verbose=False)
    # The same contexts, in a different order
    order = [3, 1, 0, 2]
    second = c2c.tensor.InteractionTensor(
        rnaseq_matrices=[matrices[i] for i in order], ppi_data=toy_ppi,
        context_names=[names[i] for i in order], how='inner', complex_sep=None,
        verbose=False)
    instance = c2c.tensor.CoupledInteractionTensor(tensor1=first, tensor2=second,
                                                   mode_mapping=SHARED_CONTEXTS,
                                                   auto_sort_shared=True)
    assert list(instance.order_names1[0]) == list(instance.order_names2[0])


# ---------------------------------------------------------------------------------
# Factorization
# ---------------------------------------------------------------------------------

def test_coupled_factorization_produces_factors(factorized_coupled):
    assert factorized_coupled.rank == 2
    assert len(factorized_coupled.factors1) == 4
    assert len(factorized_coupled.factors2) == 4
    for frame in factorized_coupled.factors1.values():
        assert list(frame.columns) == ['Factor 1', 'Factor 2']


def test_coupled_factorization_shares_the_coupled_mode(factorized_coupled):
    '''The shared dimension must have identical loadings in both tensors.'''
    label1 = factorized_coupled.order_labels1[0]
    label2 = factorized_coupled.order_labels2[0]
    shared1 = factorized_coupled.factors1[label1]
    shared2 = factorized_coupled.factors2[label2]
    assert np.allclose(shared1.values, shared2.values)


def test_coupled_factorization_is_reproducible(coupled):
    coupled.compute_tensor_factorization(rank=2, random_state=13)
    first = {k: v.copy() for k, v in coupled.factors1.items()}
    coupled.compute_tensor_factorization(rank=2, random_state=13)
    for key, value in first.items():
        pd.testing.assert_frame_equal(value, coupled.factors1[key])


def test_coupled_factorization_loadings_are_non_negative(factorized_coupled):
    for factors in [factorized_coupled.factors1, factorized_coupled.factors2]:
        for frame in factors.values():
            assert (frame.values >= -1e-9).all()


def test_coupled_factorization_indexes_match_the_element_names(factorized_coupled):
    for factors, names in [(factorized_coupled.factors1,
                            factorized_coupled.order_names1),
                           (factorized_coupled.factors2,
                            factorized_coupled.order_names2)]:
        for frame, elements in zip(factors.values(), names):
            assert list(frame.index) == list(elements)


def test_coupled_get_factorization_errors(factorized_coupled):
    errors = factorized_coupled.get_factorization_errors()
    assert errors is not None


def test_coupled_explained_variance(factorized_coupled):
    variance = factorized_coupled.explained_variance()
    assert variance is not None


def test_coupled_get_top_factor_elements(factorized_coupled):
    label = factorized_coupled.order_labels1[2]
    top = factorized_coupled.get_top_factor_elements(order_name=label,
                                                     factor_name='Factor 1',
                                                     top_number=2, tensor='tensor1')
    assert len(top) == 2
    assert top.is_monotonic_decreasing


def test_coupled_export_factor_loadings(factorized_coupled, tmp_path):
    filename = tmp_path / 'coupled.xlsx'
    factorized_coupled.export_factor_loadings(str(filename))
    assert filename.exists() and filename.stat().st_size > 0


@pytest.mark.parametrize('which', ['both', 'tensor1', 'tensor2'])
def test_coupled_fraction_helpers(factorized_coupled, which):
    '''Depending on the method and `tensor`, the result is a scalar or a dict.'''
    for method in ['excluded_value_fraction', 'sparsity_fraction', 'missing_fraction']:
        result = getattr(factorized_coupled, method)(tensor=which)
        if isinstance(result, dict):
            values = list(result.values())
        elif isinstance(result, (tuple, list)):
            values = list(result)
        else:
            values = [result]
        for value in values:
            assert 0.0 <= float(value) <= 1.0


def test_coupled_copy_is_independent(coupled):
    duplicate = coupled.copy()
    original = np.asarray(coupled.tensor1).copy()
    np.asarray(duplicate.tensor1)[0, 0, 0, 0] = 4242.0
    assert np.allclose(np.asarray(coupled.tensor1), original)


def test_coupled_write_file_roundtrip(factorized_coupled, tmp_path):
    filename = str(tmp_path / 'coupled.pkl')
    factorized_coupled.write_file(filename)
    loaded = c2c.io.load_variable_with_pickle(filename)
    assert np.allclose(np.asarray(loaded.tensor1),
                       np.asarray(factorized_coupled.tensor1))


def test_coupled_reorder_metadata(factorized_coupled):
    metadata1 = [pd.DataFrame({'Element': list(names), 'Category': list(names)})
                 for names in factorized_coupled.order_names1]
    metadata2 = [pd.DataFrame({'Element': list(names), 'Category': list(names)})
                 for names in factorized_coupled.order_names2]
    reordered = factorized_coupled.reorder_metadata(metadata1, metadata2)
    # Returns a single list covering the shared mode plus each tensor's own modes
    assert isinstance(reordered, list)
    assert len(reordered) > 0
    for frame in reordered:
        assert 'Element' in frame.columns


@pytest.mark.slow
def test_coupled_elbow_rank_selection(coupled):
    result = coupled.elbow_rank_selection(upper_rank=3, runs=1, automatic_elbow=False,
                                          manual_elbow=2, random_state=0, verbose=False)
    assert result is not None


# ---------------------------------------------------------------------------------
# coupled_non_negative_parafac
# ---------------------------------------------------------------------------------

def test_coupled_non_negative_parafac_shapes(two_tensors):
    first, second = two_tensors
    result = coupled_factorization.coupled_non_negative_parafac(
        np.asarray(first.tensor), np.asarray(second.tensor), rank=2,
        mode_mapping=SHARED_CONTEXTS, n_iter_max=10, init='random', random_state=0)
    assert result is not None


def test_coupled_non_negative_parafac_is_reproducible(two_tensors):
    first, second = two_tensors
    kwargs = dict(rank=2, mode_mapping=SHARED_CONTEXTS, n_iter_max=10, init='random',
                  random_state=4)
    one = coupled_factorization.coupled_non_negative_parafac(
        np.asarray(first.tensor), np.asarray(second.tensor), **kwargs)
    two = coupled_factorization.coupled_non_negative_parafac(
        np.asarray(first.tensor), np.asarray(second.tensor), **kwargs)
    assert str(one) == str(two)


def test_process_mode_mapping_accepts_a_dict(two_tensors):
    first, second = two_tensors
    mapping = coupled_factorization._process_mode_mapping(
        np.asarray(first.tensor), np.asarray(second.tensor), SHARED_CONTEXTS)
    assert isinstance(mapping, dict)


def test_process_mode_mapping_rejects_a_bad_type(two_tensors):
    first, second = two_tensors
    with pytest.raises(ValueError):
        coupled_factorization._process_mode_mapping(
            np.asarray(first.tensor), np.asarray(second.tensor), 'nonsense')
