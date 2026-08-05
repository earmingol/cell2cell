# -*- coding: utf-8 -*-

'''Tests for cell2cell.analysis.tensor_downstream'''

import numpy as np
import pandas as pd
import pytest

from cell2cell.analysis import tensor_downstream


LABELS = dict(sender_label='Sender Cells', receiver_label='Receiver Cells')


# ---------------------------------------------------------------------------------
# get_joint_loadings
# ---------------------------------------------------------------------------------

def test_get_joint_loadings_is_the_outer_product(factorized_tensor):
    factors = factorized_tensor.factors
    joint = tensor_downstream.get_joint_loadings(factors, dim1='Sender Cells',
                                                dim2='Receiver Cells', factor='Factor 1')
    expected = np.outer(factors['Sender Cells']['Factor 1'].values,
                        factors['Receiver Cells']['Factor 1'].values)
    assert np.allclose(joint.values, expected)
    assert list(joint.index) == list(factors['Sender Cells'].index)
    assert list(joint.columns) == list(factors['Receiver Cells'].index)


def test_get_joint_loadings_sets_axis_names(factorized_tensor):
    joint = tensor_downstream.get_joint_loadings(factorized_tensor, dim1='Sender Cells',
                                                dim2='Ligand-Receptor Pairs',
                                                factor='Factor 1')
    assert joint.index.name == 'Sender Cells'
    assert joint.columns.name == 'Ligand-Receptor Pairs'


def test_get_joint_loadings_accepts_a_tensor_or_a_dict(factorized_tensor):
    from_tensor = tensor_downstream.get_joint_loadings(factorized_tensor,
                                                       dim1='Sender Cells',
                                                       dim2='Receiver Cells',
                                                       factor='Factor 1')
    from_dict = tensor_downstream.get_joint_loadings(factorized_tensor.factors,
                                                     dim1='Sender Cells',
                                                     dim2='Receiver Cells',
                                                     factor='Factor 1')
    pd.testing.assert_frame_equal(from_tensor, from_dict)


def test_get_joint_loadings_rejects_unknown_dimension(factorized_tensor):
    with pytest.raises(AssertionError):
        tensor_downstream.get_joint_loadings(factorized_tensor, dim1='Nope',
                                             dim2='Receiver Cells', factor='Factor 1')


def test_get_joint_loadings_rejects_an_unfactorized_tensor(interaction_tensor):
    with pytest.raises(ValueError):
        tensor_downstream.get_joint_loadings(interaction_tensor, dim1='Sender Cells',
                                             dim2='Receiver Cells', factor='Factor 1')


def test_get_joint_loadings_rejects_a_bad_type():
    with pytest.raises(ValueError):
        tensor_downstream.get_joint_loadings('not-a-tensor', dim1='a', dim2='b',
                                             factor='Factor 1')


# ---------------------------------------------------------------------------------
# get_factor_specific_ccc_networks
# ---------------------------------------------------------------------------------

def test_get_factor_specific_ccc_networks_one_per_factor(factorized_tensor):
    networks = tensor_downstream.get_factor_specific_ccc_networks(factorized_tensor,
                                                                 **LABELS)
    assert list(networks.keys()) == ['Factor 1', 'Factor 2', 'Factor 3']
    cells = list(factorized_tensor.order_names[2])
    for network in networks.values():
        assert list(network.index) == cells
        assert list(network.columns) == cells


def test_networks_match_get_joint_loadings(factorized_tensor):
    networks = tensor_downstream.get_factor_specific_ccc_networks(factorized_tensor,
                                                                 **LABELS)
    for factor, network in networks.items():
        expected = tensor_downstream.get_joint_loadings(factorized_tensor,
                                                       dim1='Sender Cells',
                                                       dim2='Receiver Cells',
                                                       factor=factor)
        pd.testing.assert_frame_equal(network, expected)


def test_get_factor_specific_ccc_networks_rejects_unfactorized(interaction_tensor):
    with pytest.raises(ValueError):
        tensor_downstream.get_factor_specific_ccc_networks(interaction_tensor, **LABELS)


# ---------------------------------------------------------------------------------
# flatten_factor_ccc_networks
# ---------------------------------------------------------------------------------

def test_flatten_shape_and_labels(factorized_tensor):
    networks = tensor_downstream.get_factor_specific_ccc_networks(factorized_tensor,
                                                                 **LABELS)
    flat = tensor_downstream.flatten_factor_ccc_networks(networks)
    n_cells = len(factorized_tensor.order_names[2])
    assert flat.shape == (n_cells ** 2, 3)
    assert list(flat.columns) == list(networks.keys())
    assert all(' --> ' in label for label in flat.index)


def test_flatten_orderings_hold_the_same_values(factorized_tensor):
    networks = tensor_downstream.get_factor_specific_ccc_networks(factorized_tensor,
                                                                 **LABELS)
    by_senders = tensor_downstream.flatten_factor_ccc_networks(networks, orderby='senders')
    by_receivers = tensor_downstream.flatten_factor_ccc_networks(networks,
                                                                 orderby='receivers')
    assert set(by_senders.index) == set(by_receivers.index)
    assert list(by_senders.index) != list(by_receivers.index)
    for label in by_senders.index:
        assert np.allclose(by_senders.loc[label].values, by_receivers.loc[label].values)


# ---------------------------------------------------------------------------------
# compute_gini_coefficients
# ---------------------------------------------------------------------------------

def test_compute_gini_coefficients(factorized_tensor):
    ginis = tensor_downstream.compute_gini_coefficients(factorized_tensor, **LABELS)
    assert list(ginis.columns) == ['Factor', 'Gini']
    assert list(ginis['Factor']) == ['Factor 1', 'Factor 2', 'Factor 3']
    assert ginis['Gini'].between(0, 1).all()


def test_gini_matches_a_direct_computation(factorized_tensor):
    from cell2cell.stats import gini_coefficient
    ginis = tensor_downstream.compute_gini_coefficients(factorized_tensor, **LABELS)
    networks = tensor_downstream.get_factor_specific_ccc_networks(factorized_tensor,
                                                                 **LABELS)
    for _, row in ginis.iterrows():
        expected = gini_coefficient(networks[row['Factor']].values.flatten())
        assert np.isclose(row['Gini'], expected)


# ---------------------------------------------------------------------------------
# get_lr_by_cell_pairs
# ---------------------------------------------------------------------------------

@pytest.fixture
def downstream_kwargs():
    return dict(lr_label='Ligand-Receptor Pairs', sender_label='Sender Cells',
                receiver_label='Receiver Cells')


def test_get_lr_by_cell_pairs_shape_and_axis_names(factorized_tensor, downstream_kwargs):
    result = tensor_downstream.get_lr_by_cell_pairs(factorized_tensor, **downstream_kwargs)
    n_cells = len(factorized_tensor.order_names[2])
    assert result.shape == (len(factorized_tensor.order_names[1]), n_cells ** 2)
    assert result.columns.name == 'Sender-Receiver Pair'
    assert result.index.name == 'Ligand-Receptor Pair'


def test_get_lr_by_cell_pairs_single_factor(factorized_tensor, downstream_kwargs):
    networks = tensor_downstream.get_factor_specific_ccc_networks(factorized_tensor,
                                                                 **LABELS)
    lr_loadings = factorized_tensor.factors['Ligand-Receptor Pairs']
    result = tensor_downstream.get_lr_by_cell_pairs(factorized_tensor,
                                                    factor='Factor 2',
                                                    **downstream_kwargs)
    for cell_pair in result.columns:
        sender, receiver = cell_pair.split(' --> ')
        for lr_pair in result.index:
            expected = (networks['Factor 2'].loc[sender, receiver] *
                        lr_loadings.loc[lr_pair, 'Factor 2'])
            assert np.isclose(result.loc[lr_pair, cell_pair], expected)


def test_get_lr_by_cell_pairs_thresholds_reduce_the_output(factorized_tensor,
                                                          downstream_kwargs):
    full = tensor_downstream.get_lr_by_cell_pairs(factorized_tensor, **downstream_kwargs)
    filtered = tensor_downstream.get_lr_by_cell_pairs(factorized_tensor,
                                                      cci_threshold=0.05,
                                                      lr_threshold=0.05,
                                                      **downstream_kwargs)
    assert filtered.shape[0] <= full.shape[0]
    assert filtered.shape[1] <= full.shape[1]
    assert set(filtered.columns).issubset(set(full.columns))
    assert set(filtered.index).issubset(set(full.index))


def test_get_lr_by_cell_pairs_thresholds_keep_values_aligned(factorized_tensor,
                                                             downstream_kwargs):
    networks = tensor_downstream.get_factor_specific_ccc_networks(factorized_tensor,
                                                                 **LABELS)
    lr_loadings = factorized_tensor.factors['Ligand-Receptor Pairs']
    filtered = tensor_downstream.get_lr_by_cell_pairs(factorized_tensor,
                                                      cci_threshold=0.02,
                                                      **downstream_kwargs)
    for cell_pair in filtered.columns:
        sender, receiver = cell_pair.split(' --> ')
        for lr_pair in filtered.index:
            expected = sum(networks[f].loc[sender, receiver] * lr_loadings.loc[lr_pair, f]
                           for f in networks)
            assert np.isclose(filtered.loc[lr_pair, cell_pair], expected)


@pytest.mark.parametrize('order_cells_by', ['senders', 'receivers'])
def test_get_lr_by_cell_pairs_order_cells_by(factorized_tensor, downstream_kwargs,
                                            order_cells_by):
    result = tensor_downstream.get_lr_by_cell_pairs(factorized_tensor,
                                                    order_cells_by=order_cells_by,
                                                    **downstream_kwargs)
    n_cells = len(factorized_tensor.order_names[2])
    assert result.shape[1] == n_cells ** 2


def test_get_lr_by_cell_pairs_rejects_unknown_labels(factorized_tensor):
    with pytest.raises(AssertionError):
        tensor_downstream.get_lr_by_cell_pairs(factorized_tensor, lr_label='Nope',
                                               sender_label='Sender Cells',
                                               receiver_label='Receiver Cells')
