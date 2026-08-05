# -*- coding: utf-8 -*-

'''Tests for cell2cell.preprocessing.manipulate_dataframes'''

import numpy as np
import pandas as pd
import pytest

from cell2cell.preprocessing import manipulate_dataframes as manipulate


# ---------------------------------------------------------------------------------
# check_presence_in_dataframe
# ---------------------------------------------------------------------------------

def test_check_presence_in_dataframe_finds_elements(toy_ppi):
    found = manipulate.check_presence_in_dataframe(toy_ppi, ['Protein-A', 'Not-A-Gene'],
                                                   columns=['A', 'B'])
    assert found == ['Protein-A']


def test_check_presence_in_dataframe_defaults_to_all_columns(toy_ppi):
    found = manipulate.check_presence_in_dataframe(toy_ppi, ['Protein-F'])
    assert 'Protein-F' in found


def test_check_presence_in_dataframe_returns_nothing_when_absent(toy_ppi):
    assert manipulate.check_presence_in_dataframe(toy_ppi, ['Nope'], columns=['A']) == []


# ---------------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------------

def test_shuffle_cols_in_df_is_seeded(toy_rnaseq):
    first = manipulate.shuffle_cols_in_df(toy_rnaseq, columns=['C1'], random_state=0)
    second = manipulate.shuffle_cols_in_df(toy_rnaseq, columns=['C1'], random_state=0)
    pd.testing.assert_frame_equal(first, second)


def test_shuffle_cols_in_df_preserves_the_multiset(toy_rnaseq):
    result = manipulate.shuffle_cols_in_df(toy_rnaseq, columns=['C1'], random_state=0)
    assert sorted(result['C1'].values) == sorted(toy_rnaseq['C1'].values)
    # Untouched columns stay identical
    assert np.allclose(result['C2'].values, toy_rnaseq['C2'].values)


def test_shuffle_rows_in_df_preserves_the_multiset(toy_rnaseq):
    result = manipulate.shuffle_rows_in_df(toy_rnaseq, rows=['Protein-A'], random_state=0)
    assert sorted(result.loc['Protein-A'].values) == sorted(toy_rnaseq.loc['Protein-A'].values)
    assert np.allclose(result.loc['Protein-B'].values, toy_rnaseq.loc['Protein-B'].values)


def test_shuffle_rows_in_df_is_seeded(toy_rnaseq):
    first = manipulate.shuffle_rows_in_df(toy_rnaseq, rows=['Protein-A'], random_state=3)
    second = manipulate.shuffle_rows_in_df(toy_rnaseq, rows=['Protein-A'], random_state=3)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize('axis', [0, 1])
def test_shuffle_dataframe_preserves_shape_and_values(toy_rnaseq, axis):
    result = manipulate.shuffle_dataframe(toy_rnaseq, axis=axis, random_state=0)
    assert result.shape == toy_rnaseq.shape
    assert sorted(result.values.flatten()) == sorted(toy_rnaseq.values.flatten())
    assert list(result.index) == list(toy_rnaseq.index)
    assert list(result.columns) == list(toy_rnaseq.columns)


def test_shuffle_dataframe_is_seeded(toy_rnaseq):
    first = manipulate.shuffle_dataframe(toy_rnaseq, random_state=1)
    second = manipulate.shuffle_dataframe(toy_rnaseq, random_state=1)
    pd.testing.assert_frame_equal(first, second)


def test_shuffle_dataframe_actually_shuffles(toy_rnaseq):
    result = manipulate.shuffle_dataframe(toy_rnaseq, shuffling_number=5, random_state=0)
    assert not np.allclose(result.values, toy_rnaseq.values)


def test_shuffling_does_not_modify_input(toy_rnaseq):
    before = toy_rnaseq.copy()
    manipulate.shuffle_dataframe(toy_rnaseq, random_state=0)
    manipulate.shuffle_cols_in_df(toy_rnaseq, columns=['C1'], random_state=0)
    manipulate.shuffle_rows_in_df(toy_rnaseq, rows=['Protein-A'], random_state=0)
    pd.testing.assert_frame_equal(toy_rnaseq, before)


# ---------------------------------------------------------------------------------
# subsample_dataframe
# ---------------------------------------------------------------------------------

def test_subsample_dataframe_size_and_membership(toy_rnaseq):
    result = manipulate.subsample_dataframe(toy_rnaseq, n_samples=3, random_state=0)
    assert result.shape[0] == 3
    assert set(result.index).issubset(set(toy_rnaseq.index))


def test_subsample_dataframe_is_seeded(toy_rnaseq):
    first = manipulate.subsample_dataframe(toy_rnaseq, n_samples=3, random_state=7)
    second = manipulate.subsample_dataframe(toy_rnaseq, n_samples=3, random_state=7)
    pd.testing.assert_frame_equal(first, second)


def test_subsample_dataframe_with_full_size(toy_rnaseq):
    result = manipulate.subsample_dataframe(toy_rnaseq, n_samples=toy_rnaseq.shape[0],
                                            random_state=0)
    assert set(result.index) == set(toy_rnaseq.index)


# ---------------------------------------------------------------------------------
# check_symmetry / convert_to_distance_matrix
# ---------------------------------------------------------------------------------

def test_check_symmetry_true(toy_distance):
    assert manipulate.check_symmetry(toy_distance)


def test_check_symmetry_false(toy_distance):
    asymmetric = toy_distance.copy()
    asymmetric.iloc[0, 1] = 999.0
    assert not manipulate.check_symmetry(asymmetric)


def test_convert_to_distance_matrix_zeroes_the_diagonal(toy_distance):
    similarity = 1 - toy_distance / toy_distance.values.max()
    result = manipulate.convert_to_distance_matrix(similarity)
    assert np.allclose(np.diag(result.values), 0.0)
    assert result.shape == toy_distance.shape
