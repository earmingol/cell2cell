# -*- coding: utf-8 -*-

'''Tests for cell2cell.tensor.tensor'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.analysis.tensor_downstream import (compute_gini_coefficients,
                                                 flatten_factor_ccc_networks,
                                                 get_factor_specific_ccc_networks)


EXPECTED_LABELS = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']


# ---------------------------------------------------------------------------------
# InteractionTensor construction
# ---------------------------------------------------------------------------------

def test_interaction_tensor_shape_and_names(interaction_tensor, toy_contexts):
    tensor = interaction_tensor
    assert tensor.tensor.ndim == 4
    assert tensor.tensor.shape[0] == len(toy_contexts)
    assert len(tensor.order_names) == 4
    for names, size in zip(tensor.order_names, tensor.tensor.shape):
        assert len(names) == size


def test_default_dimension_labels_are_exposed_through_factors(interaction_tensor):
    '''Documents a known wart: `order_labels` stays None when it is not supplied.

    `compute_tensor_factorization` builds the default labels in a local variable and
    uses them as the keys of `factors`, but never assigns them back to
    `self.order_labels`. Several docstrings say the labels are "usually found in
    InteractionTensor.order_labels", so the reliable source is `factors.keys()`.
    '''
    assert interaction_tensor.order_labels is None
    interaction_tensor.compute_tensor_factorization(rank=2, random_state=0)
    assert interaction_tensor.order_labels is None
    assert list(interaction_tensor.factors.keys()) == EXPECTED_LABELS


def test_order_labels_can_be_given_upfront(toy_contexts, toy_ppi):
    labels = ['Ctx', 'LR', 'Sender', 'Receiver']
    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=list(toy_contexts.values()),
                                          ppi_data=toy_ppi,
                                          context_names=list(toy_contexts.keys()),
                                          order_labels=labels, how='inner',
                                          complex_sep=None, verbose=False)
    assert tensor.order_labels == labels


def test_interaction_tensor_uses_the_same_cells_for_both_axes(interaction_tensor):
    assert list(interaction_tensor.order_names[2]) == list(interaction_tensor.order_names[3])


def test_interaction_tensor_lr_names_use_the_caret_separator(interaction_tensor):
    for name in interaction_tensor.order_names[1]:
        assert '^' in name


def test_interaction_tensor_values_match_their_labels(toy_contexts, toy_ppi):
    '''Recomputes expression_product independently for every labelled position.'''
    matrices = list(toy_contexts.values())
    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=matrices, ppi_data=toy_ppi,
                                          context_names=list(toy_contexts.keys()),
                                          how='inner', complex_sep=None,
                                          communication_score='expression_product',
                                          verbose=False)
    # Gene names are upper-cased when building the tensor
    upper_to_original = {g.upper(): g for g in matrices[0].index}

    for c, matrix in enumerate(matrices):
        for l, lr_pair in enumerate(tensor.order_names[1]):
            ligand, receptor = lr_pair.split('^')
            for s, sender in enumerate(tensor.order_names[2]):
                for r, receiver in enumerate(tensor.order_names[3]):
                    expected = (matrix.loc[upper_to_original[ligand], sender] *
                                matrix.loc[upper_to_original[receptor], receiver])
                    assert np.isclose(tensor.tensor[c, l, s, r], expected)


@pytest.mark.parametrize('how', ['inner', 'outer', 'outer_genes', 'outer_cells'])
def test_interaction_tensor_how_options(toy_contexts, toy_ppi, how):
    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=list(toy_contexts.values()),
                                          ppi_data=toy_ppi,
                                          context_names=list(toy_contexts.keys()),
                                          how=how, outer_fraction=0.0,
                                          complex_sep=None, verbose=False)
    assert tensor.tensor.ndim == 4


def test_interaction_tensor_rejects_invalid_how(toy_contexts, toy_ppi):
    with pytest.raises(ValueError):
        c2c.tensor.InteractionTensor(rnaseq_matrices=list(toy_contexts.values()),
                                     ppi_data=toy_ppi,
                                     context_names=list(toy_contexts.keys()),
                                     how='nonsense', complex_sep=None, verbose=False)


def test_interaction_tensor_with_complexes(toy_contexts, toy_ppi_complex):
    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=list(toy_contexts.values()),
                                          ppi_data=toy_ppi_complex,
                                          context_names=list(toy_contexts.keys()),
                                          how='inner', complex_sep='&', verbose=False)
    assert tensor.tensor.ndim == 4
    assert len(tensor.order_names[1]) > 0


def test_interaction_tensor_outer_records_missing_values(toy_ppi):
    base = c2c.datasets.generate_toy_rnaseq()
    matrices = [base.drop(columns=['C5']), base.drop(columns=['C1'])]
    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=matrices, ppi_data=toy_ppi,
                                          context_names=['a', 'b'], how='outer',
                                          outer_fraction=0.0, complex_sep=None,
                                          verbose=False)
    assert tensor.mask is not None
    # Missing cells are stored as zeros and flagged in loc_nans
    assert np.asarray(tensor.loc_nans).sum() > 0


def test_interaction_tensor_excluded_value(toy_contexts, toy_ppi):
    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=list(toy_contexts.values()),
                                          ppi_data=toy_ppi,
                                          context_names=list(toy_contexts.keys()),
                                          how='inner', complex_sep=None, verbose=False)
    assert np.isfinite(np.asarray(tensor.tensor)).all()


# ---------------------------------------------------------------------------------
# PreBuiltTensor
# ---------------------------------------------------------------------------------

def test_prebuilt_tensor_labels(prebuilt_tensor):
    assert prebuilt_tensor.order_labels == EXPECTED_LABELS
    assert list(prebuilt_tensor.order_names[2]) == ['C3', 'C1', 'C2']


def test_prebuilt_tensor_default_labels():
    data = np.ones((2, 3, 4))
    tensor = c2c.tensor.PreBuiltTensor(tensor=data,
                                       order_names=[['a', 'b'], ['x', 'y', 'z'],
                                                    ['1', '2', '3', '4']])
    assert tensor.order_labels == ['Dimension-1', 'Dimension-2', 'Dimension-3']


def test_prebuilt_tensor_rejects_label_length_mismatch():
    data = np.ones((2, 2))
    with pytest.raises(AssertionError):
        c2c.tensor.PreBuiltTensor(tensor=data, order_names=[['a', 'b'], ['c', 'd']],
                                  order_labels=['only-one'])


def test_prebuilt_tensor_converts_nan_to_zero_and_records_it():
    data = np.array([[[1.0, np.nan], [2.0, 3.0]]])
    tensor = c2c.tensor.PreBuiltTensor(tensor=data,
                                       order_names=[['ctx'], ['a', 'b'], ['x', 'y']])
    assert np.asarray(tensor.tensor)[0, 0, 1] == 0.0
    assert np.asarray(tensor.loc_nans)[0, 0, 1] == 1


# ---------------------------------------------------------------------------------
# Factorization
# ---------------------------------------------------------------------------------

def test_factorization_produces_one_dataframe_per_dimension(factorized_tensor):
    factors = factorized_tensor.factors
    assert list(factors.keys()) == EXPECTED_LABELS
    for label, names in zip(EXPECTED_LABELS, factorized_tensor.order_names):
        assert list(factors[label].index) == list(names)
        assert list(factors[label].columns) == ['Factor 1', 'Factor 2', 'Factor 3']


def test_factorization_is_reproducible(interaction_tensor):
    interaction_tensor.compute_tensor_factorization(rank=2, random_state=42)
    first = {k: v.copy() for k, v in interaction_tensor.factors.items()}
    interaction_tensor.compute_tensor_factorization(rank=2, random_state=42)
    for key, value in first.items():
        pd.testing.assert_frame_equal(value, interaction_tensor.factors[key])


def test_factorization_records_the_rank_and_variance(factorized_tensor):
    assert factorized_tensor.rank == 3
    assert factorized_tensor.explained_variance_ratio_ is not None
    assert len(factorized_tensor.explained_variance_ratio_) == 3
    assert factorized_tensor.explained_variance_ is not None


def test_factorization_loadings_are_non_negative(factorized_tensor):
    for frame in factorized_tensor.factors.values():
        assert (frame.values >= -1e-9).all()


def test_factorization_variance_ordering(interaction_tensor):
    interaction_tensor.compute_tensor_factorization(rank=3, random_state=0,
                                                    var_ordered_factors=True)
    ratios = interaction_tensor.explained_variance_ratio_
    assert np.all(np.diff(ratios) <= 1e-9)


def test_get_top_factor_elements(factorized_tensor):
    top = factorized_tensor.get_top_factor_elements(order_name='Sender Cells',
                                                    factor_name='Factor 1',
                                                    top_number=2)
    assert len(top) == 2
    assert top.is_monotonic_decreasing


def test_export_factor_loadings_roundtrip(factorized_tensor, tmp_path):
    filename = tmp_path / 'loadings.xlsx'
    factorized_tensor.export_factor_loadings(str(filename))
    assert filename.exists()
    loaded = c2c.io.load_tensor_factors(str(filename))
    assert list(loaded.keys()) == EXPECTED_LABELS


@pytest.mark.slow
def test_elbow_rank_selection_runs(interaction_tensor):
    fig, errors = interaction_tensor.elbow_rank_selection(upper_rank=4, runs=1,
                                                          automatic_elbow=False,
                                                          manual_elbow=2,
                                                          random_state=0, verbose=False)
    assert len(errors) == 4
    assert interaction_tensor.rank == 2


def test_copy_is_independent(interaction_tensor):
    duplicate = interaction_tensor.copy()
    original = np.asarray(interaction_tensor.tensor).copy()
    np.asarray(duplicate.tensor)[0, 0, 0, 0] = 12345.0
    assert np.allclose(np.asarray(interaction_tensor.tensor), original)


def test_excluded_value_and_sparsity_fraction(interaction_tensor):
    fraction = interaction_tensor.sparsity_fraction()
    assert 0.0 <= fraction <= 1.0
    missing = interaction_tensor.missing_fraction()
    assert 0.0 <= missing <= 1.0


# ---------------------------------------------------------------------------------
# generate_tensor_metadata and interactions_to_tensor
# ---------------------------------------------------------------------------------

def test_generate_tensor_metadata(factorized_tensor, toy_metadata):
    cell_metadata = toy_metadata.rename(columns={'#SampleID': 'Element',
                                                 'Groups': 'Category'})
    metadata = c2c.tensor.generate_tensor_metadata(
        interaction_tensor=factorized_tensor,
        metadata_dicts=[None, None, None, None],
        fill_with_order_elements=True)
    assert len(metadata) == 4
    for frame in metadata:
        assert list(frame.columns) == ['Element', 'Category']


def test_generate_tensor_metadata_with_a_dict(factorized_tensor):
    mapping = {cell: 'group-A' for cell in factorized_tensor.order_names[2]}
    metadata = c2c.tensor.generate_tensor_metadata(
        interaction_tensor=factorized_tensor,
        metadata_dicts=[None, None, mapping, None],
        fill_with_order_elements=True)
    senders = metadata[2]
    assert set(senders['Category']) == {'group-A'}


def test_interactions_to_tensor(toy_contexts, toy_ppi):
    spaces = []
    for name, matrix in toy_contexts.items():
        space = c2c.analysis.BulkInteractions(rnaseq_data=matrix, ppi_data=toy_ppi,
                                              complex_sep=None, verbose=False)
        space.compute_pairwise_communication_scores(verbose=False)
        spaces.append(space)
    tensor = c2c.tensor.interactions_to_tensor(interactions=spaces,
                                               experiment='bulk',
                                               context_names=list(toy_contexts.keys()),
                                               how='inner', verbose=False)
    assert tensor.tensor.ndim == 4
    assert tensor.tensor.shape[0] == len(toy_contexts)


def test_interactions_to_tensor_rejects_unknown_experiment(toy_rnaseq, toy_ppi):
    space = c2c.analysis.BulkInteractions(rnaseq_data=toy_rnaseq, ppi_data=toy_ppi,
                                          complex_sep=None, verbose=False)
    space.compute_pairwise_communication_scores(verbose=False)
    with pytest.raises(ValueError):
        c2c.tensor.interactions_to_tensor(interactions=[space], experiment='nonsense',
                                          context_names=['a'], verbose=False)


# ---------------------------------------------------------------------------------
# Natural ordering of factor names
#
# The factor names were sorted lexicographically, so decompositions with 10 or more
# factors were returned as Factor 1, Factor 10, Factor 11, Factor 2, ...
# ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_factor_order_is_natural_beyond_nine_factors(interaction_tensor):
    interaction_tensor.compute_tensor_factorization(rank=12, random_state=0)
    expected = ['Factor {}'.format(i) for i in range(1, 13)]

    networks = get_factor_specific_ccc_networks(interaction_tensor)
    assert list(networks.keys()) == expected

    ginis = compute_gini_coefficients(interaction_tensor)
    assert list(ginis['Factor']) == expected

    flat = flatten_factor_ccc_networks(networks)
    assert list(flat.columns) == expected


# ---------------------------------------------------------------------------------
# Element order of the built tensor -- deliberate behaviours
#
# Guards against a future "fix" that would break them.
# ---------------------------------------------------------------------------------

def test_build_context_ccc_tensor_preserves_order_when_cells_match(toy_ppi):
    '''When every context has the same cells, the first matrix's order is preserved
    -- natural sorting is only applied when the sets differ.
    '''
    base = c2c.datasets.generate_toy_rnaseq()
    unsorted_cells = ['C5', 'C1', 'C3', 'C2', 'C4']
    matrices = [base[unsorted_cells], base[unsorted_cells] * 2.]

    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=matrices, ppi_data=toy_ppi,
                                          context_names=['a', 'b'], how='inner',
                                          complex_sep=None, verbose=False)
    assert list(tensor.order_names[2]) == unsorted_cells


def test_build_context_ccc_tensor_sorts_naturally_when_cells_differ(toy_ppi):
    base = c2c.datasets.generate_toy_rnaseq()
    renamed = base.rename(columns={'C3': 'C10', 'C4': 'C20', 'C5': 'C3'})
    renamed = renamed[['C20', 'C1', 'C10', 'C3', 'C2']]
    matrices = [renamed.drop(columns=['C20']), renamed.drop(columns=['C1'])]

    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=matrices, ppi_data=toy_ppi,
                                          context_names=['a', 'b'], how='outer',
                                          outer_fraction=0.0, complex_sep=None,
                                          verbose=False)
    cells = list(tensor.order_names[2])
    assert cells == ['C1', 'C2', 'C3', 'C10', 'C20']
    assert cells != sorted(cells)


def test_context_names_are_never_sorted(toy_contexts, toy_ppi):
    '''context_names is supplied by the user and must be preserved verbatim.'''
    names = ['Context-10', 'Context-2', 'Context-1', 'Context-3']
    tensor = c2c.tensor.InteractionTensor(rnaseq_matrices=list(toy_contexts.values()),
                                          ppi_data=toy_ppi, context_names=names,
                                          how='inner', complex_sep=None, verbose=False)
    assert list(tensor.order_names[0]) == names
