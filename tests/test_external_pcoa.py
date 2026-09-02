# -*- coding: utf-8 -*-

'''Tests for cell2cell.external.pcoa and cell2cell.external.pcoa_utils'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.external.pcoa import _fsvd
from cell2cell.external.pcoa_utils import (_e_matrix_inplace, _f_matrix_inplace,
                                           corr, e_matrix, f_matrix, mean_and_std,
                                           scale, svd_rank)


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


# ---------------------------------------------------------------------------------
# pcoa_utils: mean_and_std
# ---------------------------------------------------------------------------------

def test_mean_and_std_needs_something_to_compute():
    with pytest.raises(ValueError):
        mean_and_std(np.arange(6.), with_mean=False, with_std=False)


@pytest.mark.parametrize('axis', [None, 0, 1])
def test_mean_and_std_with_uniform_weights_matches_the_unweighted_result(axis):
    '''Uniform weights are the oracle: they must reproduce the unweighted branch.'''
    array = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 10.]])
    weights = np.ones(array.shape[0] if axis in (None, 0) else array.shape[1])
    if axis is None:
        weights = np.ones_like(array)
    plain_avg, plain_std = mean_and_std(array, axis=axis)
    weighted_avg, weighted_std = mean_and_std(array, axis=axis, weights=weights)
    assert np.allclose(plain_avg, weighted_avg)
    assert np.allclose(plain_std, weighted_std)


def test_mean_and_std_weights_shift_the_average():
    array = np.array([[0., 0.], [10., 10.]])
    avg, _ = mean_and_std(array, axis=0, weights=np.array([3., 1.]))
    assert np.allclose(avg, [2.5, 2.5])


@pytest.mark.parametrize('axis', [None, 0])
def test_mean_and_std_rescales_the_variance_for_ddof(axis):
    '''ddof != 0 multiplies the variance by n / (n - ddof).'''
    array = np.array([[1., 2.], [3., 5.], [6., 9.]])
    weights = np.ones_like(array) if axis is None else np.ones(array.shape[0])
    _, std_0 = mean_and_std(array, axis=axis, weights=weights, ddof=0)
    _, std_1 = mean_and_std(array, axis=axis, weights=weights, ddof=1)
    n = array.size if axis is None else array.shape[axis]
    assert np.allclose(std_1 ** 2, std_0 ** 2 * n / (n - 1))


def test_mean_and_std_can_skip_the_mean():
    array = np.array([[1., 2.], [3., 5.]])
    avg, std = mean_and_std(array, axis=0, weights=np.ones(2), with_mean=False)
    assert avg is None
    assert std is not None


def test_mean_and_std_can_skip_the_std():
    array = np.array([[1., 2.], [3., 5.]])
    avg, std = mean_and_std(array, axis=0, weights=np.ones(2), with_std=False)
    assert std is None
    assert avg is not None


# ---------------------------------------------------------------------------------
# pcoa_utils: corr, svd_rank and the in-place centering helpers
# ---------------------------------------------------------------------------------

def test_corr_of_a_matrix_with_itself_has_a_unit_diagonal():
    x = np.array([[1., 4.], [2., 1.], [3., 7.], [4., 2.]])
    result = corr(x)
    assert result.shape == (2, 2)
    assert np.allclose(np.diag(result), 1.0)


def test_corr_between_two_matrices_detects_a_perfect_relationship():
    x = np.array([[1.], [2.], [3.], [4.]])
    result = corr(x, 2 * x + 5)
    assert result.shape == (1, 1)
    assert np.isclose(result[0, 0], 1.0)


def test_corr_rejects_matrices_with_different_row_counts():
    with pytest.raises(ValueError):
        corr(np.ones((4, 2)), np.ones((3, 2)))


def test_svd_rank_counts_the_non_negligible_singular_values():
    # A rank-2 matrix: the third singular value is numerically zero.
    matrix = np.outer([1., 2., 3.], [1., 0., 1.]) + np.outer([0., 1., 0.], [0., 1., 0.])
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert svd_rank(matrix.shape, singular_values) == 2


def test_svd_rank_honours_an_explicit_tolerance():
    singular_values = np.array([5., 1., 0.1])
    assert svd_rank((3, 3), singular_values, tol=0.5) == 2


def test_inplace_centering_matches_the_copying_version(toy_distance):
    '''The non-in-place `e_matrix`/`f_matrix` pair is the oracle for the in-place one.'''
    array = np.array(toy_distance, dtype=float)
    expected = f_matrix(e_matrix(array))
    centered = _f_matrix_inplace(_e_matrix_inplace(array.copy()))
    assert np.allclose(centered, expected)


def test_e_matrix_inplace_squares_and_halves_negatively(toy_distance):
    array = np.array(toy_distance, dtype=float)
    assert np.allclose(_e_matrix_inplace(array.copy()), e_matrix(array))


# ---------------------------------------------------------------------------------
# pcoa: the fsvd method and the argument checks
# ---------------------------------------------------------------------------------

def test_pcoa_with_fsvd_returns_the_requested_dimensions(toy_distance):
    result = c2c.external.pcoa(toy_distance, method='fsvd', number_of_dimensions=2)
    assert result['samples'].shape == (toy_distance.shape[0], 2)
    assert list(result['samples'].index) == list(toy_distance.index)


def test_pcoa_with_fsvd_approximates_the_exact_decomposition(toy_distance):
    '''FSVD draws a random Gaussian matrix, so only a loose agreement is expected.'''
    exact = c2c.external.pcoa(toy_distance, method='eigh', number_of_dimensions=2)
    approximate = c2c.external.pcoa(toy_distance, method='fsvd', number_of_dimensions=2)
    assert np.allclose(np.asarray(exact['eigvals'])[:2],
                       np.asarray(approximate['eigvals'])[:2], rtol=1e-3, atol=1e-6)


def test_pcoa_with_fsvd_and_all_dimensions(toy_distance):
    '''`number_of_dimensions=0` asks for every dimension.'''
    result = c2c.external.pcoa(toy_distance, method='fsvd', number_of_dimensions=0)
    assert result['samples'].shape[0] == toy_distance.shape[0]


def test_pcoa_rejects_an_unknown_method(toy_distance):
    with pytest.raises(ValueError):
        c2c.external.pcoa(toy_distance, method='nonsense')


def test_pcoa_rejects_a_negative_number_of_dimensions(toy_distance):
    with pytest.raises(ValueError):
        c2c.external.pcoa(toy_distance, number_of_dimensions=-1)


def test_fsvd_rejects_a_non_square_matrix():
    with pytest.raises(ValueError):
        _fsvd(np.ones((3, 4)), 2)


def test_fsvd_rejects_more_dimensions_than_the_matrix_has():
    with pytest.raises(ValueError):
        _fsvd(np.ones((3, 3)), 5)


def test_fsvd_rejects_a_negative_number_of_dimensions():
    with pytest.raises(ValueError):
        _fsvd(np.ones((3, 3)), -1)
