# -*- coding: utf-8 -*-

'''Tests for cell2cell.tensor.external_scores'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.tensor.external_scores import _ordered_intersection


TENSOR_KWARGS = dict(sender_col='source', receiver_col='target', ligand_col='ligand',
                     receptor_col='receptor', score_col='score')


@pytest.fixture
def context_dict(toy_liana):
    return {name: frame for name, frame in toy_liana.groupby('context')}


def test_ordered_intersection_keeps_first_list_order():
    lists = [['c', 'a', 'b'], ['b', 'c'], ['c', 'b', 'z']]
    assert _ordered_intersection(lists) == ['c', 'b']


def test_ordered_intersection_of_identical_lists():
    assert _ordered_intersection([['x', 'y'], ['x', 'y']]) == ['x', 'y']


def test_dataframes_to_tensor_shape_and_labels(context_dict):
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, how='inner', **TENSOR_KWARGS)
    assert tensor.tensor.ndim == 4
    contexts, lr_pairs, senders, receivers = [list(o) for o in tensor.order_names]
    assert len(contexts) == 3
    assert len(lr_pairs) == 10
    assert senders == receivers == ['CT-1', 'CT-2', 'CT-3']


def test_dataframes_to_tensor_values_match_labels(context_dict):
    '''Independently recompute every labelled position from the long dataframe.'''
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, how='inner', **TENSOR_KWARGS)
    contexts, lr_pairs, senders, receivers = [list(o) for o in tensor.order_names]

    for c, context in enumerate(contexts):
        frame = context_dict[context]
        for l, lr_pair in enumerate(lr_pairs):
            ligand, receptor = lr_pair.split('^')
            for s, sender in enumerate(senders):
                for r, receiver in enumerate(receivers):
                    match = frame[(frame['source'] == sender) &
                                  (frame['target'] == receiver) &
                                  (frame['ligand'] == ligand) &
                                  (frame['receptor'] == receptor)]
                    assert np.isclose(tensor.tensor[c, l, s, r], match['score'].iloc[0])


def test_dataframes_to_tensor_sorts_elements_naturally():
    liana = c2c.datasets.generate_toy_liana_output(n_contexts=11, n_cell_types=11)
    context_dict = {name: frame for name, frame in liana.groupby('context')}
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, how='inner',
                                            sort_elements=True, **TENSOR_KWARGS)
    contexts = list(tensor.order_names[0])
    senders = list(tensor.order_names[2])
    assert contexts[-2:] == ['Context-10', 'Context-11']
    assert contexts != sorted(contexts)
    assert senders[-2:] == ['CT-10', 'CT-11']


def test_dataframes_to_tensor_without_sorting_is_reproducible(context_dict):
    first = c2c.tensor.dataframes_to_tensor(context_dict, how='inner',
                                           sort_elements=False, **TENSOR_KWARGS)
    second = c2c.tensor.dataframes_to_tensor(context_dict, how='inner',
                                            sort_elements=False, **TENSOR_KWARGS)
    assert [list(o) for o in first.order_names] == [list(o) for o in second.order_names]
    assert np.allclose(np.asarray(first.tensor), np.asarray(second.tensor))


def test_dataframes_to_tensor_unsorted_still_aligns_values(context_dict):
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, how='inner',
                                            sort_elements=False, **TENSOR_KWARGS)
    contexts, lr_pairs, senders, receivers = [list(o) for o in tensor.order_names]
    frame = context_dict[contexts[0]]
    ligand, receptor = lr_pairs[0].split('^')
    match = frame[(frame['source'] == senders[0]) & (frame['target'] == receivers[0]) &
                  (frame['ligand'] == ligand) & (frame['receptor'] == receptor)]
    assert np.isclose(tensor.tensor[0, 0, 0, 0], match['score'].iloc[0])


def test_dataframes_to_tensor_respects_a_given_context_order(context_dict):
    order = ['Context-3', 'Context-1', 'Context-2']
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, context_order=order,
                                             how='inner', **TENSOR_KWARGS)
    assert list(tensor.order_names[0]) == order


def test_dataframes_to_tensor_rejects_bad_context_order(context_dict):
    with pytest.raises(AssertionError):
        c2c.tensor.dataframes_to_tensor(context_dict, context_order=['nope'],
                                        how='inner', **TENSOR_KWARGS)


def test_dataframes_to_tensor_rejects_missing_columns(context_dict):
    broken = {k: v.drop(columns=['score']) for k, v in context_dict.items()}
    with pytest.raises(AssertionError):
        c2c.tensor.dataframes_to_tensor(broken, how='inner', **TENSOR_KWARGS)


def test_dataframes_to_tensor_rejects_invalid_how(context_dict):
    with pytest.raises(ValueError):
        c2c.tensor.dataframes_to_tensor(context_dict, how='nonsense', **TENSOR_KWARGS)


@pytest.mark.parametrize('how', ['inner', 'outer', 'outer_lrs', 'outer_cells'])
def test_dataframes_to_tensor_how_options(context_dict, how):
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, how=how, outer_fraction=0.0,
                                            **TENSOR_KWARGS)
    assert tensor.tensor.ndim == 4


def test_dataframes_to_tensor_outer_flags_absent_combinations(toy_liana):
    '''Missing combinations become 0.0 and are recorded in loc_nans.'''
    liana = toy_liana[~((toy_liana['context'] == 'Context-1') &
                        (toy_liana['source'] == 'CT-3'))]
    context_dict = {name: frame for name, frame in liana.groupby('context')}
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, how='outer', outer_fraction=0.0,
                                            **TENSOR_KWARGS)
    loc_nans = np.asarray(tensor.loc_nans)
    assert loc_nans.sum() > 0
    # Where flagged as missing, the stored value must be zero
    assert np.allclose(np.asarray(tensor.tensor)[loc_nans == 1], 0.0)


def test_dataframes_to_tensor_aggregates_duplicates(context_dict):
    duplicated = {k: pd.concat([v, v.head(1)]) for k, v in context_dict.items()}
    tensor = c2c.tensor.dataframes_to_tensor(duplicated, how='inner',
                                            dup_aggregation='max', **TENSOR_KWARGS)
    assert tensor.tensor.ndim == 4


def test_dataframes_to_tensor_custom_lr_separator(context_dict):
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, how='inner', lr_sep='::',
                                             **TENSOR_KWARGS)
    assert all('::' in name for name in tensor.order_names[1])


def test_dataframes_to_tensor_custom_order_labels(context_dict):
    labels = ['Samples', 'LRs', 'From', 'To']
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, how='inner',
                                             order_labels=labels, **TENSOR_KWARGS)
    assert tensor.order_labels == labels


def test_dataframes_to_tensor_can_be_factorized(context_dict):
    tensor = c2c.tensor.dataframes_to_tensor(context_dict, how='inner', **TENSOR_KWARGS)
    tensor.compute_tensor_factorization(rank=2, random_state=0)
    assert len(tensor.factors) == 4
