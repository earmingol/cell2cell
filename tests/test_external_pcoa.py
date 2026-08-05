# -*- coding: utf-8 -*-

'''Tests for cell2cell.external.pcoa and cell2cell.external.pcoa_utils'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.external.pcoa_utils import scale


# ---------------------------------------------------------------------------------
# pcoa
# ---------------------------------------------------------------------------------

def test_pcoa_returns_the_expected_keys(toy_distance):
    result = c2c.external.pcoa(toy_distance)
    for key in ['samples', 'eigvals', 'proportion_explained']:
        assert key in result


def test_pcoa_sample_shape_and_labels(toy_distance):
    result = c2c.external.pcoa(toy_distance)
    assert result['samples'].shape[0] == toy_distance.shape[0]
    assert list(result['samples'].index) == list(toy_distance.index)


def test_pcoa_proportion_explained_is_a_distribution(toy_distance):
    result = c2c.external.pcoa(toy_distance)
    proportions = np.asarray(result['proportion_explained'])
    assert (proportions >= -1e-9).all()
    assert np.isclose(proportions.sum(), 1.0, atol=1e-6)


def test_pcoa_eigenvalues_are_sorted_descending(toy_distance):
    result = c2c.external.pcoa(toy_distance)
    eigvals = np.asarray(result['eigvals'])
    assert np.all(np.diff(eigvals) <= 1e-9)


def test_pcoa_with_a_limited_number_of_dimensions(toy_distance):
    result = c2c.external.pcoa(toy_distance, number_of_dimensions=2)
    assert result['samples'].shape[1] == 2


def test_pcoa_rejects_asymmetric_input(toy_distance):
    asymmetric = toy_distance.copy()
    asymmetric.iloc[0, 1] = 42.0
    with pytest.raises(ValueError):
        c2c.external.pcoa(asymmetric)


def test_pcoa_is_deterministic(toy_distance):
    first = c2c.external.pcoa(toy_distance)
    second = c2c.external.pcoa(toy_distance)
    assert np.allclose(first['samples'].values, second['samples'].values)


def test_pcoa_biplot_runs(toy_distance, toy_rnaseq):
    ordination = c2c.external.pcoa(toy_distance)
    # pcoa_biplot expects the samples frame, not the whole result dictionary
    features = toy_rnaseq.T
    result = c2c.external.pcoa_biplot(ordination, features)
    assert 'features' in result
    assert list(result['features'].index) == list(features.columns)


def test_check_ordination_accepts_a_pcoa_result(toy_distance):
    ordination = c2c.external.pcoa(toy_distance)
    checked = c2c.external._check_ordination(ordination)
    assert checked is not None


# ---------------------------------------------------------------------------------
# convert_to_distance_matrix raised instead of warning
#
# `raise Warning(...)` aborts, so the diagonal was never "automatically replaced by
# zeros" as the message claimed. This broke the public `pcoa()` for any similarity
# or correlation matrix, since pcoa calls it unconditionally.
# ---------------------------------------------------------------------------------

def test_pcoa_accepts_a_similarity_matrix(toy_distance):
    '''pcoa() raised Warning for any matrix whose diagonal was not already zero.'''
    similarity = 1 - toy_distance / toy_distance.values.max()
    with pytest.warns(UserWarning):
        result = c2c.external.pcoa(similarity)
    assert result['samples'].shape[0] == toy_distance.shape[0]


# ---------------------------------------------------------------------------------
# pcoa_biplot recursed through pandas
#
# `np.power(eigvals, -0.5, where=...)` was called on a pandas Series, which recurses
# in pandas' __array_ufunc__ handling. The missing `out=` also left the masked
# entries reading uninitialized memory.
# ---------------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------------
# Arrays that cannot be written to
#
# pandas >= 3.0 enforces copy-on-write, so `DataFrame.values` returns a read-only
# array. `pcoa(inplace=True)` was dead code on top of that, since it went through
# `np.float`, an alias numpy removed in 1.24.
# ---------------------------------------------------------------------------------

def test_pcoa_centers_the_matrix_in_place_on_request(toy_distance):
    inplace = c2c.external.pcoa(toy_distance, inplace=True)
    default = c2c.external.pcoa(toy_distance, inplace=False)
    assert np.allclose(inplace['eigvals'].values, default['eigvals'].values)
    assert np.allclose(np.abs(inplace['samples'].values),
                       np.abs(default['samples'].values))


def test_pcoa_does_not_modify_its_input(toy_distance):
    before = toy_distance.copy()
    c2c.external.pcoa(toy_distance, inplace=True)
    pd.testing.assert_frame_equal(toy_distance, before)


def test_pcoa_biplot_accepts_a_read_only_frame(read_only_frame, toy_distance, toy_rnaseq):
    distance = read_only_frame(toy_distance, labels=list(toy_distance.index))
    ordination = c2c.external.pcoa(distance)
    features = toy_rnaseq.T
    result = c2c.external.pcoa_biplot(ordination, features)
    assert list(result['features'].index) == list(features.columns)


# ---------------------------------------------------------------------------------
# pcoa_utils
# ---------------------------------------------------------------------------------

def test_scale_accepts_a_dataframe():
    '''`scale` standardizes its argument in place. It used to copy before converting to an
    array, so a dataframe was copied as a dataframe and the conversion then handed back
    the read-only buffer that pandas >= 3.0 exposes. Arrays were never affected.'''
    frame = pd.DataFrame([[1., 2.], [3., 4.], [5., 7.]], columns=['x', 'y'])
    result = scale(frame)
    assert np.allclose(result.mean(axis=0), 0.0)
    assert np.allclose(result.std(axis=0), 1.0)


def test_scale_does_not_modify_its_input():
    array = np.array([[1., 2.], [3., 4.], [5., 7.]])
    before = array.copy()
    scale(array)
    assert np.allclose(array, before)
