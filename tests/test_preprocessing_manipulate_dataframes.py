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


@pytest.mark.parametrize('dtype', [object, 'string'])
def test_check_presence_in_dataframe_accepts_a_single_column_name(toy_ppi, dtype):
    '''The documented argument is a list, but a bare column name is normalized instead of
    being handed to pandas as-is, which would return a Series. Both the object and the
    "string" dtype are covered, since pandas >= 3.0 makes the latter the default.'''
    ppi_data = toy_ppi.copy()
    ppi_data[['A', 'B']] = ppi_data[['A', 'B']].astype(dtype)
    found = manipulate.check_presence_in_dataframe(ppi_data, ['Protein-A'], columns='A')
    assert found == ['Protein-A']


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


# ---------------------------------------------------------------------------------
# zero_diagonal
# ---------------------------------------------------------------------------------

def test_zero_diagonal_zeroes_the_diagonal_and_keeps_the_rest(toy_distance):
    similarity = 1 - toy_distance / toy_distance.values.max()
    result = manipulate.zero_diagonal(similarity)
    assert np.allclose(np.diag(result.values), 0.0)
    # Off-diagonal values are untouched
    off_diagonal = ~np.eye(similarity.shape[0], dtype=bool)
    assert np.allclose(result.values[off_diagonal], similarity.values[off_diagonal])


def test_zero_diagonal_keeps_the_labels(toy_distance):
    result = manipulate.zero_diagonal(toy_distance)
    assert list(result.index) == list(toy_distance.index)
    assert list(result.columns) == list(toy_distance.columns)


def test_zero_diagonal_does_not_modify_its_input(toy_distance):
    similarity = 1 - toy_distance / toy_distance.values.max()
    before = similarity.copy()
    manipulate.zero_diagonal(similarity)
    pd.testing.assert_frame_equal(similarity, before)


# ---------------------------------------------------------------------------------
# Dataframes whose `.values` array is read-only
#
# pandas >= 3.0 enforces copy-on-write, so `DataFrame.values` returns a read-only array
# and `np.fill_diagonal`/`np.random.shuffle` on it raise "underlying array is read-only".
# A `df.copy()` does not help, since the copy's `.values` is read-only as well.
# ---------------------------------------------------------------------------------

def test_zero_diagonal_accepts_a_read_only_frame(read_only_frame, toy_distance):
    similarity = read_only_frame(1 - toy_distance / toy_distance.values.max(),
                                 labels=list(toy_distance.index))
    result = manipulate.zero_diagonal(similarity)
    assert np.allclose(np.diag(result.values), 0.0)


def test_convert_to_distance_matrix_accepts_a_read_only_frame(read_only_frame, toy_distance):
    similarity = read_only_frame(1 - toy_distance / toy_distance.values.max(),
                                 labels=list(toy_distance.index))
    with pytest.warns(UserWarning):
        result = manipulate.convert_to_distance_matrix(similarity)
    assert np.allclose(np.diag(result.values), 0.0)


def test_shuffle_dataframe_accepts_a_read_only_frame(read_only_frame, toy_rnaseq):
    frame = read_only_frame(toy_rnaseq)
    result = manipulate.shuffle_dataframe(frame, random_state=0)
    assert result.shape == frame.shape
    assert sorted(result.values.flatten()) == sorted(frame.values.flatten())


# ---------------------------------------------------------------------------------
# convert_to_distance_matrix raised instead of warning
#
# `raise Warning(...)` aborts, so the diagonal was never "automatically replaced by
# zeros" as the message claimed. This broke the public `pcoa()` for any similarity
# or correlation matrix, since pcoa calls it unconditionally.
# ---------------------------------------------------------------------------------

def test_convert_to_distance_matrix_replaces_a_non_zero_diagonal(toy_distance):
    similarity = 1 - toy_distance / toy_distance.values.max()
    assert not np.allclose(np.diag(similarity.values), 0.0)

    with pytest.warns(UserWarning):
        result = manipulate.convert_to_distance_matrix(similarity)
    assert np.allclose(np.diag(result.values), 0.0)


def test_convert_to_distance_matrix_still_rejects_asymmetric_input(toy_distance):
    asymmetric = toy_distance.copy()
    asymmetric.iloc[0, 1] = 999.0
    with pytest.raises(ValueError):
        manipulate.convert_to_distance_matrix(asymmetric)


# ---------------------------------------------------------------------------------
# check_presence_in_dataframe crashed on mixed data types
#
# It sorted the values with np.unique, which cannot compare strings to floats, so
# the documented `columns=None` default failed on any dataframe holding both.
# ---------------------------------------------------------------------------------

def test_check_presence_in_dataframe_handles_mixed_dtypes(toy_ppi):
    # toy_ppi mixes gene names with a float 'score' column
    assert toy_ppi.dtypes.nunique() > 1
    found = manipulate.check_presence_in_dataframe(toy_ppi, ['Protein-F'])
    assert found == ['Protein-F']
