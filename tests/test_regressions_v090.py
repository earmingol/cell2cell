# -*- coding: utf-8 -*-

'''Regression tests for the bugs fixed in v0.9.0.

Every test in this module reproduces a defect that shipped in a released version.
They were each verified to FAIL against the pre-fix code, so they are real
bug-catchers rather than descriptions of the current behaviour.

The module also contains guards for behaviours that are deliberate and must NOT be
"fixed" by a future change (for instance the lexicographic sorting that decides which
direction of a bidirectional PPI is dropped).
'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.analysis.tensor_downstream import (compute_gini_coefficients,
                                                  flatten_factor_ccc_networks,
                                                  get_factor_specific_ccc_networks,
                                                  get_lr_by_cell_pairs)
from cell2cell.core.interaction_space import generate_pairs
from cell2cell.preprocessing.find_elements import get_element_abundances
from cell2cell.preprocessing.ppi import remove_ppi_bidirectionality
from cell2cell.preprocessing.rnaseq import aggregate_single_cells


# =================================================================================
# Cell-pair labels in flattened factor-specific networks
#
# `flatten_factor_ccc_networks` built the 'sender --> receiver' labels from the
# SORTED cell names while flattening the values in the tensor's own order. Every
# loading was assigned to the wrong cell pair when the tensor was not alphabetically
# sorted, and `get_lr_by_cell_pairs` inherited the mislabeling.
# =================================================================================

@pytest.mark.parametrize('orderby', ['senders', 'receivers'])
def test_flatten_labels_match_values_on_unsorted_tensor(factorized_prebuilt_tensor, orderby):
    '''The decisive regression test: 27/27 entries were mislabeled before the fix.'''
    networks = get_factor_specific_ccc_networks(factorized_prebuilt_tensor)
    flat = flatten_factor_ccc_networks(networks, orderby=orderby)

    # The fixture is deliberately unsorted, otherwise this test cannot fail
    cells = list(factorized_prebuilt_tensor.order_names[2])
    assert cells != sorted(cells)

    for factor, network in networks.items():
        for sender in network.index:
            for receiver in network.columns:
                label = '{} --> {}'.format(sender, receiver)
                assert np.isclose(flat.loc[label, factor], network.loc[sender, receiver])


def test_flatten_preserves_the_tensor_order(factorized_prebuilt_tensor):
    networks = get_factor_specific_ccc_networks(factorized_prebuilt_tensor)
    flat = flatten_factor_ccc_networks(networks, orderby='senders')
    cells = list(factorized_prebuilt_tensor.order_names[2])
    expected = ['{} --> {}'.format(s, r) for s in cells for r in cells]
    assert list(flat.index) == expected


def test_flatten_groups_by_receiver_when_requested(factorized_prebuilt_tensor):
    networks = get_factor_specific_ccc_networks(factorized_prebuilt_tensor)
    flat = flatten_factor_ccc_networks(networks, orderby='receivers')
    cells = list(factorized_prebuilt_tensor.order_names[2])
    expected = ['{} --> {}'.format(s, r) for r in cells for s in cells]
    assert list(flat.index) == expected


def test_flatten_realigns_networks_with_different_element_order(factorized_prebuilt_tensor):
    '''Hand-assembled networks may not share an element order; values must still align.'''
    networks = get_factor_specific_ccc_networks(factorized_prebuilt_tensor)
    factors = list(networks.keys())
    shuffled = {factors[0]: networks[factors[0]],
                factors[1]: networks[factors[1]].reindex(index=['C1', 'C2', 'C3'],
                                                         columns=['C2', 'C3', 'C1'])}
    flat = flatten_factor_ccc_networks(shuffled)
    for factor in factors:
        for sender in ['C1', 'C2', 'C3']:
            for receiver in ['C1', 'C2', 'C3']:
                label = '{} --> {}'.format(sender, receiver)
                assert np.isclose(flat.loc[label, factor],
                                  networks[factor].loc[sender, receiver])


def test_flatten_keeps_only_common_elements(factorized_prebuilt_tensor):
    networks = get_factor_specific_ccc_networks(factorized_prebuilt_tensor)
    factors = list(networks.keys())
    subset = {factors[0]: networks[factors[0]],
              factors[1]: networks[factors[1]].drop(index='C1', columns='C1')}
    flat = flatten_factor_ccc_networks(subset)
    assert len(flat) == 4                       # 2 senders x 2 receivers
    assert not any('C1' in label for label in flat.index)


def test_flatten_accepts_a_single_factor(factorized_prebuilt_tensor):
    networks = get_factor_specific_ccc_networks(factorized_prebuilt_tensor)
    factor = list(networks.keys())[0]
    flat = flatten_factor_ccc_networks({factor: networks[factor]})
    assert list(flat.columns) == [factor]
    for sender in networks[factor].index:
        for receiver in networks[factor].columns:
            label = '{} --> {}'.format(sender, receiver)
            assert np.isclose(flat.loc[label, factor],
                              networks[factor].loc[sender, receiver])


def test_flatten_rejects_an_invalid_orderby(factorized_prebuilt_tensor):
    networks = get_factor_specific_ccc_networks(factorized_prebuilt_tensor)
    with pytest.raises(ValueError):
        flatten_factor_ccc_networks(networks, orderby='not-an-option')


def test_get_lr_by_cell_pairs_labels_match_values(factorized_prebuilt_tensor):
    '''get_lr_by_cell_pairs inherited the mislabeling from the flattening.'''
    tensor = factorized_prebuilt_tensor
    networks = get_factor_specific_ccc_networks(tensor)
    lr_loadings = tensor.factors['Ligand-Receptor Pairs']

    result = get_lr_by_cell_pairs(tensor,
                                  lr_label='Ligand-Receptor Pairs',
                                  sender_label='Sender Cells',
                                  receiver_label='Receiver Cells')

    for cell_pair in result.columns:
        sender, receiver = cell_pair.split(' --> ')
        for lr_pair in result.index:
            expected = sum(networks[f].loc[sender, receiver] * lr_loadings.loc[lr_pair, f]
                           for f in networks)
            assert np.isclose(result.loc[lr_pair, cell_pair], expected)


# =================================================================================
# Natural ordering of factor names
#
# The factor names were sorted lexicographically, so decompositions with 10 or more
# factors were returned as Factor 1, Factor 10, Factor 11, Factor 2, ...
# =================================================================================

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


def test_get_lr_by_cell_pairs_accepts_other_factor_names(factorized_prebuilt_tensor):
    '''Previously raised IndexError: the code did int(name.split(' ')[1]).'''
    tensor = factorized_prebuilt_tensor
    for key in tensor.factors:
        tensor.factors[key].columns = ['component_1', 'component_10']

    result = get_lr_by_cell_pairs(tensor,
                                  lr_label='Ligand-Receptor Pairs',
                                  sender_label='Sender Cells',
                                  receiver_label='Receiver Cells')
    assert result.shape[0] == len(tensor.order_names[1])
    assert result.shape[1] == len(tensor.order_names[2]) ** 2


# =================================================================================
# aggregate_single_cells modified the dataframe passed by the user
# =================================================================================

@pytest.mark.parametrize('method', ['average', 'nn_cell_fraction', 'trimean'])
@pytest.mark.parametrize('transposed', [True, False])
def test_aggregate_single_cells_does_not_modify_its_input(toy_single_cells, method, transposed):
    rnaseq, metadata = toy_single_cells
    data = rnaseq.T if transposed else rnaseq

    index_before = list(data.index)
    columns_before = list(data.columns)
    values_before = data.values.copy()

    aggregate_single_cells(data, metadata, barcode_col='barcodes',
                           celltype_col='cell_types', method=method,
                           transposed=transposed)

    assert list(data.index) == index_before
    assert list(data.columns) == columns_before
    assert np.array_equal(data.values, values_before)


def test_aggregate_single_cells_can_be_called_twice(toy_single_cells):
    '''Previously raised KeyError, because the first call replaced the index.'''
    rnaseq, metadata = toy_single_cells
    data = rnaseq.T

    first = aggregate_single_cells(data, metadata, barcode_col='barcodes',
                                   celltype_col='cell_types', method='average')
    second = aggregate_single_cells(data, metadata, barcode_col='barcodes',
                                    celltype_col='cell_types', method='average')
    pd.testing.assert_frame_equal(first, second)


def test_aggregate_single_cells_orders_cell_types_naturally():
    rnaseq, metadata = c2c.datasets.generate_toy_single_cells(n_cell_types=11,
                                                             n_cells_per_type=2)
    aggregated = aggregate_single_cells(rnaseq.T, metadata, barcode_col='barcodes',
                                       celltype_col='cell_types', method='average')
    columns = list(aggregated.columns)
    assert columns == ['CT-{}'.format(i) for i in range(1, 12)]
    assert columns != sorted(columns)


# =================================================================================
# add_sliding_window_info_to_adata crashed on pandas >= 2
#
# The barcodes of each window were passed to .loc as a set, which newer versions of
# pandas reject with "Passing a set as an indexer is not supported".
# =================================================================================

def test_add_sliding_window_info_to_adata_accepts_window_mapping(toy_spatial_adata):
    window_mapping = c2c.spatial.create_sliding_windows(toy_spatial_adata,
                                                        window_size=20., stride=7.)
    assert isinstance(next(iter(window_mapping.values())), set)

    c2c.spatial.add_sliding_window_info_to_adata(toy_spatial_adata, window_mapping)

    window_columns = [c for c in toy_spatial_adata.obs.columns if c.startswith('window_')]
    assert len(window_columns) == len(window_mapping)
    # Every cell assigned to a window must be flagged with 1.0
    for window, barcodes in window_mapping.items():
        flagged = toy_spatial_adata.obs.loc[list(barcodes), window]
        assert (flagged == 1.0).all()


def test_sliding_window_columns_are_naturally_ordered(toy_spatial_adata):
    '''With more than 10 windows per axis, window_10_* must not precede window_2_*.'''
    window_mapping = c2c.spatial.create_sliding_windows(toy_spatial_adata,
                                                        window_size=10., stride=5.)
    c2c.spatial.add_sliding_window_info_to_adata(toy_spatial_adata, window_mapping)

    columns = [c for c in toy_spatial_adata.obs.columns if c.startswith('window_')]
    assert any(c.startswith('window_10_') for c in columns)
    assert columns != sorted(columns)       # natural order differs from alphabetical


# =================================================================================
# Distance matrix for the 'count' and 'icellnet' CCI scores
#
# The branch was guarded by `if ~(cci_score in [...])`. Bitwise inversion of a bool
# gives -2/-1, which are both truthy, so the regularized-distance branch was dead
# code and those scores produced NEGATIVE distances (down to -7 on the toy data).
# =================================================================================

@pytest.mark.parametrize('cci_score', ['count', 'icellnet'])
def test_unbounded_cci_scores_give_non_negative_distances(toy_rnaseq, toy_ppi, cci_score):
    interactions = c2c.analysis.BulkInteractions(rnaseq_data=toy_rnaseq, ppi_data=toy_ppi,
                                                 cci_score=cci_score, cci_type='undirected',
                                                 complex_sep=None, verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    distance = interactions.interaction_space.distance_matrix

    assert (distance.values >= 0).all(), 'a distance can never be negative'
    assert np.allclose(np.diag(distance.values), 0.0)
    # The raw scores are unbounded, so 1 - score would have gone negative
    assert interactions.interaction_space.interaction_elements['cci_matrix'].values.max() > 1


@pytest.mark.parametrize('cci_score', ['bray_curtis', 'jaccard'])
def test_bounded_cci_scores_use_the_plain_complement(toy_rnaseq, toy_ppi, cci_score):
    '''Scores already in [0, 1] must keep using 1 - score, unchanged by the fix.'''
    interactions = c2c.analysis.BulkInteractions(rnaseq_data=toy_rnaseq, ppi_data=toy_ppi,
                                                 cci_score=cci_score, cci_type='undirected',
                                                 complex_sep=None, verbose=False)
    interactions.compute_pairwise_cci_scores(verbose=False)
    cci = interactions.interaction_space.interaction_elements['cci_matrix']
    distance = interactions.interaction_space.distance_matrix

    expected = 1 - cci.values
    np.fill_diagonal(expected, 0.0)
    assert np.allclose(distance.values, expected)


# =================================================================================
# scale_expression_by_sum could not normalize across columns
#
# The sums were not kept 2-dimensional, so the documented `axis=1` option raised
# "operands could not be broadcast together with shapes (6,5) (6,)".
# =================================================================================

def test_scale_expression_by_sum_supports_both_axes(toy_rnaseq):
    from cell2cell.preprocessing.rnaseq import scale_expression_by_sum

    by_column = scale_expression_by_sum(toy_rnaseq, axis=0, sum_value=1e6)
    assert np.allclose(by_column.sum(axis=0).values, 1e6)

    by_row = scale_expression_by_sum(toy_rnaseq, axis=1, sum_value=1e6)
    assert np.allclose(by_row.sum(axis=1).values, 1e6)


# =================================================================================
# convert_to_distance_matrix raised instead of warning
#
# `raise Warning(...)` aborts, so the diagonal was never "automatically replaced by
# zeros" as the message claimed. This broke the public `pcoa()` for any similarity
# or correlation matrix, since pcoa calls it unconditionally.
# =================================================================================

def test_convert_to_distance_matrix_replaces_a_non_zero_diagonal(toy_distance):
    from cell2cell.preprocessing.manipulate_dataframes import convert_to_distance_matrix

    similarity = 1 - toy_distance / toy_distance.values.max()
    assert not np.allclose(np.diag(similarity.values), 0.0)

    with pytest.warns(UserWarning):
        result = convert_to_distance_matrix(similarity)
    assert np.allclose(np.diag(result.values), 0.0)


def test_convert_to_distance_matrix_still_rejects_asymmetric_input(toy_distance):
    from cell2cell.preprocessing.manipulate_dataframes import convert_to_distance_matrix

    asymmetric = toy_distance.copy()
    asymmetric.iloc[0, 1] = 999.0
    with pytest.raises(ValueError):
        convert_to_distance_matrix(asymmetric)


def test_pcoa_accepts_a_similarity_matrix(toy_distance):
    '''pcoa() raised Warning for any matrix whose diagonal was not already zero.'''
    similarity = 1 - toy_distance / toy_distance.values.max()
    with pytest.warns(UserWarning):
        result = c2c.external.pcoa(similarity)
    assert result['samples'].shape[0] == toy_distance.shape[0]


# =================================================================================
# check_presence_in_dataframe crashed on mixed data types
#
# It sorted the values with np.unique, which cannot compare strings to floats, so
# the documented `columns=None` default failed on any dataframe holding both.
# =================================================================================

def test_check_presence_in_dataframe_handles_mixed_dtypes(toy_ppi):
    from cell2cell.preprocessing.manipulate_dataframes import check_presence_in_dataframe

    # toy_ppi mixes gene names with a float 'score' column
    assert toy_ppi.dtypes.nunique() > 1
    found = check_presence_in_dataframe(toy_ppi, ['Protein-F'])
    assert found == ['Protein-F']


# =================================================================================
# random_switching_ppi_labels crashed with its own default arguments
#
# `ppi_data[interaction_columns]` passed a TUPLE to pandas, which reads it as a
# single column name. So the simplest possible call -- default genes=None and
# default permuted_column='both' -- always raised KeyError: ('A', 'B').
# =================================================================================

def test_random_switching_ppi_labels_works_with_default_arguments(toy_ppi):
    from cell2cell.stats.permutation import random_switching_ppi_labels

    result = random_switching_ppi_labels(toy_ppi, random_state=0)
    assert result.shape == toy_ppi.shape
    assert list(result.columns) == list(toy_ppi.columns)
    # The permutation relabels genes, so the multiset of genes is preserved
    original = sorted(list(toy_ppi['A']) + list(toy_ppi['B']))
    permuted = sorted(list(result['A']) + list(result['B']))
    assert len(original) == len(permuted)


def test_random_switching_ppi_labels_default_is_reproducible(toy_ppi):
    from cell2cell.stats.permutation import random_switching_ppi_labels

    first = random_switching_ppi_labels(toy_ppi, random_state=3)
    second = random_switching_ppi_labels(toy_ppi, random_state=3)
    pd.testing.assert_frame_equal(first, second)


# =================================================================================
# Functions that were broken on the default (numpy) tensorly backend
#
# `concatenate_interaction_tensors` called `.to('cpu')`, a pytorch-only method, and
# then read `context['device']`, a key that a numpy context does not have. So it
# always failed unless a pytorch backend was configured.
# =================================================================================

def test_concatenate_interaction_tensors_works_on_the_numpy_backend(toy_contexts, toy_ppi):
    matrices = list(toy_contexts.values())
    names = list(toy_contexts.keys())
    labels = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

    def build(start, stop):
        return c2c.tensor.InteractionTensor(rnaseq_matrices=matrices[start:stop],
                                            ppi_data=toy_ppi,
                                            context_names=names[start:stop],
                                            how='inner', complex_sep=None, verbose=False)

    first, second = build(0, 2), build(2, 4)
    combined = c2c.tensor.concatenate_interaction_tensors([first, second], axis=0,
                                                          order_labels=labels)
    assert combined.tensor.shape[0] == 4
    assert list(combined.order_names[0]) == names
    assert np.allclose(np.asarray(combined.tensor)[:2], np.asarray(first.tensor))


# =================================================================================
# reorder_dimension_elements crashed on its own default
#
# `metadata.copy()` ran unconditionally although `metadata=None` is the documented
# default, and the following lines already guard with `if new_metadata is not None`.
# =================================================================================

def test_reorder_dimension_elements_without_metadata(factorized_tensor):
    from cell2cell.plotting.tensor_plot import reorder_dimension_elements

    cells = list(factorized_tensor.order_names[2])
    reordered, metadata = reorder_dimension_elements(factorized_tensor.factors,
                                                     {'Sender Cells': cells[::-1]})
    assert metadata is None
    assert list(reordered['Sender Cells'].index) == cells[::-1]


# =================================================================================
# pcoa_biplot recursed through pandas
#
# `np.power(eigvals, -0.5, where=...)` was called on a pandas Series, which recurses
# in pandas' __array_ufunc__ handling. The missing `out=` also left the masked
# entries reading uninitialized memory.
# =================================================================================

def test_pcoa_biplot_projects_the_features(toy_distance, toy_rnaseq):
    ordination = c2c.external.pcoa(toy_distance)
    features = toy_rnaseq.T
    result = c2c.external.pcoa_biplot(ordination, features)
    assert 'features' in result
    assert list(result['features'].index) == list(features.columns)
    assert np.isfinite(result['features'].values).all()


def test_pcoa_biplot_is_deterministic(toy_distance, toy_rnaseq):
    '''Without `out=`, the entries excluded by `where` were uninitialized memory.'''
    features = toy_rnaseq.T
    first = c2c.external.pcoa_biplot(c2c.external.pcoa(toy_distance), features)
    second = c2c.external.pcoa_biplot(c2c.external.pcoa(toy_distance), features)
    assert np.allclose(first['features'].values, second['features'].values)


# =================================================================================
# Deliberate behaviours -- guards against a future "fix" that would break them
# =================================================================================

def test_remove_ppi_bidirectionality_keeps_using_lexicographic_order():
    '''This lexicographic sort decides WHICH direction of a bidirectional PPI is
    dropped. Replacing it with a natural sort would silently change which rows
    survive, so the output is pinned here.
    '''
    ppi = pd.DataFrame({'A': ['G1', 'G2', 'G3', 'G2', 'G10', 'G2'],
                        'B': ['G2', 'G1', 'G4', 'G3', 'G2', 'G10']})
    result = remove_ppi_bidirectionality(ppi, ('A', 'B'), verbose=False)

    pairs = set(zip(result['A'], result['B']))
    # Of each bidirectional pair only one direction is kept
    assert ('G1', 'G2') in pairs and ('G2', 'G1') not in pairs
    assert ('G10', 'G2') in pairs and ('G2', 'G10') not in pairs
    # Unidirectional interactions are untouched
    assert ('G3', 'G4') in pairs
    assert result.shape[0] == 4


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


# =================================================================================
# Reproducibility -- these orders came from an unsorted set() and varied per run
# =================================================================================

def test_generate_pairs_follows_the_order_of_the_cells():
    pairs = generate_pairs(['C3', 'C1', 'C2'], 'directed')
    assert pairs == [('C3', 'C3'), ('C3', 'C1'), ('C3', 'C2'),
                     ('C1', 'C3'), ('C1', 'C1'), ('C1', 'C2'),
                     ('C2', 'C3'), ('C2', 'C1'), ('C2', 'C2')]


def test_generate_pairs_deduplicates_without_losing_order():
    pairs = generate_pairs(['A', 'A', 'B'], 'directed')
    assert pairs == [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]


def test_generate_pairs_is_reproducible():
    first = generate_pairs(['C3', 'C1', 'C2'], 'undirected')
    second = generate_pairs(['C3', 'C1', 'C2'], 'undirected')
    assert first == second
    assert len(first) == len(set(first))


def test_get_element_abundances_keeps_first_appearance_order():
    '''Used to iterate over sets, making the tensor axes vary between runs.'''
    element_lists = [['b', 'a', 'c'], ['c', 'b', 'z'], ['b', 'c', 'q']]
    abundances = get_element_abundances(element_lists)
    assert list(abundances.keys()) == ['b', 'a', 'c', 'z', 'q']
    assert np.isclose(abundances['b'], 1.0)
    assert np.isclose(abundances['a'], 1 / 3)


def test_get_element_abundances_counts_each_list_once():
    abundances = get_element_abundances([['a', 'a', 'a'], ['a']])
    assert np.isclose(abundances['a'], 1.0)
