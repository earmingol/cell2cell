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


# ---------------------------------------------------------------------------------
# _process_mode_mapping: the non-dict forms kept for backward compatibility
# ---------------------------------------------------------------------------------

def _cube(seed=0):
    return np.random.default_rng(seed).random((3, 3, 3))


def test_process_mode_mapping_accepts_an_int():
    '''An int names the one mode that is NOT shared, so the rest are paired up.'''
    mapping = coupled_factorization._process_mode_mapping(_cube(), _cube(1), 2)
    assert mapping == {'shared': [(0, 0), (1, 1)]}


@pytest.mark.parametrize('non_shared', [[1, 2], (1, 2)])
def test_process_mode_mapping_accepts_a_list_or_tuple(non_shared):
    mapping = coupled_factorization._process_mode_mapping(_cube(), _cube(1), non_shared)
    assert mapping == {'shared': [(0, 0)]}


def test_process_mode_mapping_deduplicates_repeated_modes():
    mapping = coupled_factorization._process_mode_mapping(_cube(), _cube(1), [2, 2])
    assert mapping == {'shared': [(0, 0), (1, 1)]}


def test_process_mode_mapping_needs_a_dict_for_tensors_of_different_order():
    with pytest.raises(ValueError):
        coupled_factorization._process_mode_mapping(_cube(), np.ones((3, 3)), 1)


# ---------------------------------------------------------------------------------
# _validate_tensors
# ---------------------------------------------------------------------------------

def test_validate_tensors_accepts_compatible_raw_tensors():
    # Returns None; the point is that it does not raise.
    assert coupled_factorization._validate_tensors(_cube(), _cube(1), SHARED_CONTEXTS) is None


@pytest.mark.parametrize('mapping', [{'shared': [(9, 0)]},        # bad tensor1 mode
                                     {'shared': [(0, 9)]},        # bad tensor2 mode
                                     {'shared': []}])             # nothing shared
def test_validate_tensors_rejects_bad_mode_mappings(mapping):
    with pytest.raises(ValueError):
        coupled_factorization._validate_tensors(_cube(), _cube(1), mapping)


def test_validate_tensors_rejects_shared_modes_of_different_size():
    with pytest.raises(ValueError):
        coupled_factorization._validate_tensors(_cube(), np.ones((4, 3, 3)), SHARED_CONTEXTS)


def test_validate_tensors_rejects_mismatched_element_names(two_tensors):
    first, second = two_tensors
    second.order_names[0] = ['Other-' + name for name in second.order_names[0]]
    with pytest.raises(ValueError):
        coupled_factorization._validate_tensors(first, second, SHARED_CONTEXTS)


# ---------------------------------------------------------------------------------
# _compute_balancing_weights
# ---------------------------------------------------------------------------------

def test_balancing_weights_uses_manual_weights_when_given():
    weights = coupled_factorization._compute_balancing_weights(
        _cube(), _cube(1), SHARED_CONTEXTS, manual_weights=(2.0, 1.0))
    assert weights == (2.0, 1.0)


def test_balancing_weights_are_equal_when_balancing_is_off():
    weights = coupled_factorization._compute_balancing_weights(
        _cube(), _cube(1), SHARED_CONTEXTS, balance_errors=False)
    assert weights == (0.5, 0.5)


def test_balancing_weights_are_inversely_proportional_to_the_non_shared_size():
    '''Without manual weights, the smaller tensor gets the larger weight.'''
    small = np.ones((3, 2, 2))    # non-shared size 4
    large = np.ones((3, 4, 4))    # non-shared size 16
    weight1, weight2 = coupled_factorization._compute_balancing_weights(
        small, large, SHARED_CONTEXTS, manual_weights=None)
    assert weight1 > weight2
    assert np.isclose(weight1, 20 / 4)
    assert np.isclose(weight2, 20 / 16)


@pytest.mark.parametrize('manual_weights', [(1.0, 1.0, 1.0), 'nonsense', (0.0, 1.0), (1.0, -2.0)])
def test_balancing_weights_reject_invalid_manual_weights(manual_weights):
    with pytest.raises(ValueError):
        coupled_factorization._compute_balancing_weights(
            _cube(), _cube(1), SHARED_CONTEXTS, manual_weights=manual_weights)


# ---------------------------------------------------------------------------------
# _create_combined_factors_dict
# ---------------------------------------------------------------------------------

def test_create_combined_factors_dict_keeps_one_copy_of_the_shared_mode():
    factors1 = {0: np.ones((3, 2)), 1: np.full((4, 2), 2.)}
    factors2 = {0: np.ones((3, 2)) * 9, 1: np.full((5, 2), 3.)}
    combined = coupled_factorization._create_combined_factors_dict(
        factors1, factors2, SHARED_CONTEXTS)
    assert len(combined) == 3
    # The shared mode comes from tensor1, not tensor2
    assert np.allclose(combined[0], factors1[0])
    assert combined[1].shape == (4, 2)
    assert combined[2].shape == (5, 2)


# ---------------------------------------------------------------------------------
# coupled_non_negative_parafac options
# ---------------------------------------------------------------------------------

def test_coupled_non_negative_parafac_rejects_an_unknown_convergence_criterion(two_tensors):
    first, second = two_tensors
    with pytest.raises(ValueError):
        coupled_factorization.coupled_non_negative_parafac(
            np.asarray(first.tensor), np.asarray(second.tensor), rank=2,
            mode_mapping=SHARED_CONTEXTS, n_iter_max=5, init='random', random_state=0,
            cvg_criterion='nonsense')


def test_coupled_non_negative_parafac_can_return_the_errors(two_tensors):
    first, second = two_tensors
    cp1, cp2, (errors1, errors2) = coupled_factorization.coupled_non_negative_parafac(
        np.asarray(first.tensor), np.asarray(second.tensor), rank=2,
        mode_mapping=SHARED_CONTEXTS, n_iter_max=5, init='random', random_state=0,
        return_errors=True)
    assert len(errors1) == len(errors2)
    assert len(errors1) >= 1


def test_coupled_non_negative_parafac_with_masks(two_tensors):
    '''A mask marks values to ignore, and must not change the shape of the factors.'''
    first, second = two_tensors
    tensor1 = np.asarray(first.tensor)
    tensor2 = np.asarray(second.tensor)
    mask1 = np.ones(tensor1.shape)
    mask1[0, 0, 0, 0] = 0
    mask2 = np.ones(tensor2.shape)
    mask2[0, 0, 0, 0] = 0
    cp1, cp2 = coupled_factorization.coupled_non_negative_parafac(
        tensor1, tensor2, rank=2, mode_mapping=SHARED_CONTEXTS, n_iter_max=5,
        init='random', random_state=0, mask1=mask1, mask2=mask2)
    assert [f.shape for f in cp1.factors] == [(d, 2) for d in tensor1.shape]
    assert [f.shape for f in cp2.factors] == [(d, 2) for d in tensor2.shape]


@pytest.mark.parametrize('separate_weights', [True, False])
def test_coupled_non_negative_parafac_normalizes_its_factors(two_tensors, separate_weights):
    first, second = two_tensors
    cp1, cp2 = coupled_factorization.coupled_non_negative_parafac(
        np.asarray(first.tensor), np.asarray(second.tensor), rank=2,
        mode_mapping=SHARED_CONTEXTS, n_iter_max=5, init='random', random_state=0,
        normalize_factors=True, separate_weights=separate_weights)
    for factor in cp1.factors + cp2.factors:
        assert np.allclose(np.linalg.norm(factor, axis=0), 1.0)


# ---------------------------------------------------------------------------------
# CoupledInteractionTensor: the option branches
# ---------------------------------------------------------------------------------

def test_coupled_get_factorization_errors_can_plot(factorized_coupled):
    errors, fig = factorized_coupled.get_factorization_errors(plot=True)
    assert fig is not None


def test_coupled_get_factorization_errors_before_factorizing(coupled):
    assert coupled.get_factorization_errors() is None


def test_coupled_to_device_falls_back_on_an_unusable_device(factorized_coupled):
    '''The numpy backend has no devices, so the request is caught and ignored.'''
    factorized_coupled.to_device('not-a-device')
    assert factorized_coupled.tensor1 is not None
    assert factorized_coupled.tensor2 is not None


def test_coupled_export_factor_loadings_separately(factorized_coupled, tmp_path):
    filename = tmp_path / 'loadings.xlsx'
    factorized_coupled.export_factor_loadings(str(filename), save_separate=True)
    assert list(tmp_path.glob('*.xlsx'))


def test_coupled_factor_dataframes_without_normalizing(coupled):
    coupled.compute_tensor_factorization(rank=2, random_state=0, normalize_loadings=False)
    assert coupled.factors is not None
    assert all(frame.shape[1] == 2 for frame in coupled.factors.values())


@pytest.mark.parametrize('fraction', ['excluded_value_fraction', 'sparsity_fraction',
                                      'missing_fraction'])
@pytest.mark.parametrize('which', ['tensor1', 'tensor2'])
def test_coupled_fraction_helpers_per_tensor(factorized_coupled, fraction, which):
    value = getattr(factorized_coupled, fraction)(tensor=which)
    assert 0.0 <= float(value) <= 1.0


def test_coupled_excluded_value_fraction_without_a_mask(factorized_coupled):
    factorized_coupled.mask1 = None
    factorized_coupled.mask2 = None
    assert factorized_coupled.excluded_value_fraction() == {'tensor1': 0.0, 'tensor2': 0.0}


@pytest.mark.parametrize('fraction', ['excluded_value_fraction', 'sparsity_fraction',
                                      'missing_fraction'])
def test_coupled_fraction_helpers_combine_both_tensors(factorized_coupled, fraction):
    '''"combined" is a size-weighted average, so it lies between the two tensors.'''
    per_tensor = getattr(factorized_coupled, fraction)(tensor='both')
    combined = getattr(factorized_coupled, fraction)(tensor='combined')
    assert min(per_tensor.values()) - 1e-9 <= combined <= max(per_tensor.values()) + 1e-9


@pytest.mark.parametrize('fraction', ['excluded_value_fraction', 'sparsity_fraction',
                                      'missing_fraction'])
def test_coupled_fraction_helpers_reject_an_unknown_tensor(factorized_coupled, fraction):
    with pytest.raises(ValueError):
        getattr(factorized_coupled, fraction)(tensor='nonsense')


# ---------------------------------------------------------------------------------
# elbow_rank_selection, kept small on purpose
# ---------------------------------------------------------------------------------

SMALL_ELBOW = dict(upper_rank=2, n_iter_max=10, tol=1e-3, random_state=0, verbose=False)


@pytest.mark.slow
def test_coupled_elbow_with_multiple_runs(coupled):
    '''runs > 1 routes through `plot_multiple_run_coupled_elbow`.'''
    fig, loss = coupled.elbow_rank_selection(runs=2, automatic_elbow=False, manual_elbow=1,
                                             show_individual=True, **SMALL_ELBOW)
    assert fig is not None
    assert len(loss['combined']) == 2


@pytest.mark.slow
def test_coupled_elbow_picks_a_rank_automatically(coupled, monkeypatch):
    '''On a curve this short kneed often finds no knee at all, and the rank is then
    read straight off `_compute_elbow`, so the detector is pinned rather than run.'''
    monkeypatch.setattr(c2c.tensor.coupled_tensor, '_compute_elbow', lambda loss: 2)
    coupled.elbow_rank_selection(runs=1, automatic_elbow=True, output_fig=False,
                                 **SMALL_ELBOW)
    assert coupled.rank == 2
    assert coupled.elbow_metric == 'error'


@pytest.mark.slow
def test_coupled_elbow_smooths_the_curve(coupled, monkeypatch):
    monkeypatch.setattr(c2c.tensor.coupled_tensor, '_compute_elbow', lambda loss: 1)
    monkeypatch.setattr(c2c.tensor.coupled_tensor, 'smooth_curve', lambda values: values)
    fig, loss = coupled.elbow_rank_selection(runs=2, smooth=True, output_fig=False,
                                             **SMALL_ELBOW)
    assert len(loss['combined']) == 2


@pytest.mark.slow
def test_coupled_elbow_with_the_similarity_metric(coupled):
    fig, loss = coupled.elbow_rank_selection(runs=2, metric='similarity',
                                             automatic_elbow=False, manual_elbow=1,
                                             output_fig=False, **SMALL_ELBOW)
    assert coupled.elbow_metric == 'similarity'
    assert fig is None
