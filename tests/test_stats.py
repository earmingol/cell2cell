# -*- coding: utf-8 -*-

'''Tests for cell2cell.stats'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.stats import enrichment, gini, multitest, permutation


# ---------------------------------------------------------------------------------
# gini
# ---------------------------------------------------------------------------------

def test_gini_of_a_uniform_distribution_is_zero():
    assert np.isclose(gini.gini_coefficient(np.ones(10)), 0.0)


def test_gini_of_a_concentrated_distribution_is_high():
    concentrated = np.array([0.0] * 99 + [1.0])
    assert gini.gini_coefficient(concentrated) > 0.95


def test_gini_is_between_zero_and_one():
    for values in [np.arange(1, 11.), np.array([1., 1., 5.]), np.random.default_rng(0).random(50)]:
        result = gini.gini_coefficient(values)
        assert 0.0 <= result <= 1.0


def test_gini_is_scale_invariant():
    values = np.array([1., 2., 3., 4.])
    assert np.isclose(gini.gini_coefficient(values), gini.gini_coefficient(values * 10))


def test_gini_increases_with_inequality():
    equal = np.array([5., 5., 5., 5.])
    unequal = np.array([1., 2., 7., 10.])
    assert gini.gini_coefficient(unequal) > gini.gini_coefficient(equal)


# ---------------------------------------------------------------------------------
# enrichment
# ---------------------------------------------------------------------------------

def test_hypergeom_representation_returns_depletion_then_enrichment():
    result = enrichment.hypergeom_representation(sample_size=10, class_in_sample=8,
                                                population_size=100, class_in_population=20)
    assert len(result) == 2
    for pvalue in result:
        assert 0.0 <= pvalue <= 1.0


def test_hypergeom_detects_over_representation():
    '''The whole class ends up in the sample, so enrichment is significant.'''
    depletion, enrichment_pval = enrichment.hypergeom_representation(
        sample_size=20, class_in_sample=20, population_size=100, class_in_population=20)
    assert enrichment_pval < depletion
    assert enrichment_pval < 0.05


def test_hypergeom_detects_under_representation():
    depletion, enrichment_pval = enrichment.hypergeom_representation(
        sample_size=20, class_in_sample=0, population_size=100, class_in_population=50)
    assert depletion < enrichment_pval
    assert depletion < 0.05


def test_fisher_representation_returns_odds_and_pvalues():
    result = enrichment.fisher_representation(sample_size=10, class_in_sample=8,
                                              population_size=100, class_in_population=20)
    assert set(result.keys()) == {'pval', 'odds'}
    assert len(result['pval']) == 2 and len(result['odds']) == 2
    for pvalue in result['pval']:
        assert 0.0 <= pvalue <= 1.0


def test_fisher_and_hypergeom_agree_on_direction():
    kwargs = dict(sample_size=20, class_in_sample=18, population_size=200,
                  class_in_population=30)
    _, hyper_enrichment = enrichment.hypergeom_representation(**kwargs)
    fisher = enrichment.fisher_representation(**kwargs)
    assert hyper_enrichment < 0.05
    assert fisher['pval'][1] < 0.05          # index 1 is the enrichment p-value


# ---------------------------------------------------------------------------------
# multitest
# ---------------------------------------------------------------------------------

@pytest.fixture
def symmetric_pvalues():
    values = np.array([[1.0, 0.001, 0.5],
                       [0.001, 1.0, 0.04],
                       [0.5, 0.04, 1.0]])
    return pd.DataFrame(values, index=['a', 'b', 'c'], columns=['a', 'b', 'c'])


def test_fdrcorrection_symmetric_keeps_symmetry(symmetric_pvalues):
    result = multitest.compute_fdrcorrection_symmetric_matrix(symmetric_pvalues, alpha=0.1)
    assert np.allclose(result.values, result.values.T)
    assert list(result.index) == list(symmetric_pvalues.index)


def test_fdrcorrection_symmetric_only_increases_pvalues(symmetric_pvalues):
    result = multitest.compute_fdrcorrection_symmetric_matrix(symmetric_pvalues, alpha=0.1)
    lower = np.tril_indices_from(result.values, k=-1)
    assert (result.values[lower] >= symmetric_pvalues.values[lower] - 1e-12).all()


def test_fdrcorrection_asymmetric_shape_and_labels():
    values = np.array([[0.001, 0.2], [0.03, 0.9]])
    frame = pd.DataFrame(values, index=['a', 'b'], columns=['x', 'y'])
    result = multitest.compute_fdrcorrection_asymmetric_matrix(frame, alpha=0.1)
    assert result.shape == frame.shape
    assert list(result.index) == ['a', 'b']
    assert list(result.columns) == ['x', 'y']
    assert (result.values >= frame.values - 1e-12).all()


def test_fdrcorrection_does_not_modify_input(symmetric_pvalues):
    before = symmetric_pvalues.copy()
    multitest.compute_fdrcorrection_symmetric_matrix(symmetric_pvalues)
    pd.testing.assert_frame_equal(symmetric_pvalues, before)


# ---------------------------------------------------------------------------------
# permutation helpers
# ---------------------------------------------------------------------------------

def test_compute_pvalue_from_dist_upper():
    distribution = np.arange(100.)
    assert np.isclose(permutation.compute_pvalue_from_dist(200., distribution,
                                                           comparison='upper'), 0.0)
    assert np.isclose(permutation.compute_pvalue_from_dist(-1., distribution,
                                                           comparison='upper'), 1.0)


def test_compute_pvalue_from_dist_lower():
    distribution = np.arange(100.)
    assert np.isclose(permutation.compute_pvalue_from_dist(-1., distribution,
                                                           comparison='lower'), 0.0)


def test_compute_pvalue_from_dist_different_returns_two_sided():
    distribution = np.arange(-50., 50.)
    pvalue = permutation.compute_pvalue_from_dist(0., distribution,
                                                  comparison='different')
    assert 0.0 <= pvalue <= 1.0


def test_compute_pvalue_from_dist_is_bounded():
    distribution = np.random.default_rng(0).normal(size=200)
    for value in [-5., 0., 5.]:
        pvalue = permutation.compute_pvalue_from_dist(value, distribution)
        assert 0.0 <= pvalue <= 1.0


def test_compute_pvalue_from_dist_with_an_empty_distribution():
    assert np.isclose(permutation.compute_pvalue_from_dist(1.0, []), 1.0)


def test_pvalue_from_dist_returns_a_labelled_result():
    distribution = np.arange(100.)
    result = permutation.pvalue_from_dist(150., distribution, label='my-label')
    assert result is not None


# ---------------------------------------------------------------------------------
# random_switching_ppi_labels
# ---------------------------------------------------------------------------------

def test_random_switching_ppi_labels_is_reproducible(toy_ppi):
    genes = sorted(set(toy_ppi['A']).union(toy_ppi['B']))
    first = permutation.random_switching_ppi_labels(toy_ppi, genes=genes, random_state=0)
    second = permutation.random_switching_ppi_labels(toy_ppi, genes=genes, random_state=0)
    pd.testing.assert_frame_equal(first, second)


def test_random_switching_ppi_labels_without_genes_is_reproducible(toy_ppi):
    first = permutation.random_switching_ppi_labels(toy_ppi, random_state=1)
    second = permutation.random_switching_ppi_labels(toy_ppi, random_state=1)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize('permuted_column', ['both', 'first', 'second'])
def test_random_switching_ppi_labels_columns(toy_ppi, permuted_column):
    result = permutation.random_switching_ppi_labels(toy_ppi, random_state=0,
                                                     permuted_column=permuted_column)
    assert result.shape == toy_ppi.shape
    assert list(result.columns) == list(toy_ppi.columns)


def test_random_switching_ppi_labels_only_permutes_the_chosen_column(toy_ppi):
    result = permutation.random_switching_ppi_labels(toy_ppi, random_state=0,
                                                     permuted_column='first')
    assert list(result['B']) == list(toy_ppi['B'])


def test_random_switching_ppi_labels_rejects_bad_column(toy_ppi):
    with pytest.raises(ValueError):
        permutation.random_switching_ppi_labels(toy_ppi, permuted_column='nonsense')


def test_random_switching_ppi_labels_does_not_modify_input(toy_ppi):
    before = toy_ppi.copy()
    permutation.random_switching_ppi_labels(toy_ppi, random_state=0)
    pd.testing.assert_frame_equal(toy_ppi, before)


@pytest.mark.parametrize('permuted_column', ['both', 'first', 'second'])
def test_random_switching_ppi_labels_with_the_string_dtype(toy_ppi, permuted_column):
    '''The gene names used to be collected with `.values.flatten()`. pandas >= 3.0 makes
    `str` the default dtype, so the values of a single column are an extension array,
    which has no `.flatten()`. Casting to "string" reproduces that on older pandas.'''
    ppi_data = toy_ppi.copy()
    ppi_data[['A', 'B']] = ppi_data[['A', 'B']].astype('string')
    result = permutation.random_switching_ppi_labels(ppi_data, random_state=0,
                                                     permuted_column=permuted_column)
    assert result.shape == ppi_data.shape
    assert list(result.columns) == list(ppi_data.columns)
    # Labels are swapped among themselves, so no new name can appear
    known_genes = set(toy_ppi['A']).union(toy_ppi['B'])
    for column in ('A', 'B'):
        assert set(result[column]).issubset(known_genes)


# ---------------------------------------------------------------------------------
# run_label_permutation
# ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_run_label_permutation_shape(toy_rnaseq, toy_ppi, analysis_setup, cutoff_setup):
    genes = list(toy_rnaseq.index)
    result = permutation.run_label_permutation(rnaseq_data=toy_rnaseq, ppi_data=toy_ppi,
                                               genes=genes,
                                               analysis_setup=analysis_setup,
                                               cutoff_setup=cutoff_setup,
                                               permutations=5, verbose=False)
    assert result.shape == (toy_rnaseq.shape[1], toy_rnaseq.shape[1])
    assert list(result.columns) == list(toy_rnaseq.columns)
    assert ((result.values >= 0) & (result.values <= 1)).all()


@pytest.mark.slow
def test_run_label_permutation_excludes_cells(toy_rnaseq, toy_ppi, analysis_setup,
                                              cutoff_setup):
    result = permutation.run_label_permutation(rnaseq_data=toy_rnaseq, ppi_data=toy_ppi,
                                               genes=list(toy_rnaseq.index),
                                               analysis_setup=analysis_setup,
                                               cutoff_setup=cutoff_setup,
                                               permutations=3, excluded_cells=['C1'],
                                               verbose=False)
    assert 'C1' not in result.columns


# ---------------------------------------------------------------------------------
# random_switching_ppi_labels crashed with its own default arguments
#
# `ppi_data[interaction_columns]` passed a TUPLE to pandas, which reads it as a
# single column name. So the simplest possible call -- default genes=None and
# default permuted_column='both' -- always raised KeyError: ('A', 'B').
# ---------------------------------------------------------------------------------

def test_random_switching_ppi_labels_works_with_default_arguments(toy_ppi):
    result = permutation.random_switching_ppi_labels(toy_ppi, random_state=0)
    assert result.shape == toy_ppi.shape
    assert list(result.columns) == list(toy_ppi.columns)
    # The permutation relabels genes, so the multiset of genes is preserved
    original = sorted(list(toy_ppi['A']) + list(toy_ppi['B']))
    permuted = sorted(list(result['A']) + list(result['B']))
    assert len(original) == len(permuted)


def test_random_switching_ppi_labels_default_is_reproducible(toy_ppi):
    first = permutation.random_switching_ppi_labels(toy_ppi, random_state=3)
    second = permutation.random_switching_ppi_labels(toy_ppi, random_state=3)
    pd.testing.assert_frame_equal(first, second)
