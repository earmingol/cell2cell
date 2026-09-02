# -*- coding: utf-8 -*-

'''Tests for cell2cell.core.cci_scores'''

import numpy as np
import pandas as pd
import pytest

from cell2cell.core import cci_scores
from cell2cell.core.cci_scores import matmul_cosine


class FakeCell:
    '''Minimal stand-in exposing only what the scoring functions read.'''

    def __init__(self, a_values, b_values):
        self.weighted_ppi = pd.DataFrame({'A': a_values, 'B': b_values})


PAIRWISE = [cci_scores.compute_jaccard_like_cci_score,
            cci_scores.compute_braycurtis_like_cci_score,
            cci_scores.compute_count_score,
            cci_scores.compute_icellnet_score]


@pytest.mark.parametrize('score_function', PAIRWISE)
def test_scores_are_zero_for_empty_cells(score_function):
    empty = FakeCell([], [])
    assert score_function(empty, empty) == 0.0


@pytest.mark.parametrize('score_function', PAIRWISE)
def test_scores_are_zero_when_nothing_is_expressed(score_function):
    silent = FakeCell([0.0, 0.0], [0.0, 0.0])
    assert score_function(silent, silent) == 0.0


@pytest.mark.parametrize('score_function', [cci_scores.compute_jaccard_like_cci_score,
                                            cci_scores.compute_braycurtis_like_cci_score])
def test_bounded_scores_stay_within_zero_and_one(score_function):
    cell1 = FakeCell([1.0, 0.5, 0.0], [0.2, 1.0, 0.7])
    cell2 = FakeCell([0.3, 0.9, 1.0], [1.0, 0.1, 0.4])
    score = score_function(cell1, cell2)
    assert 0.0 <= score <= 1.0


def test_jaccard_like_of_identical_binary_vectors_is_one():
    cell = FakeCell([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert np.isclose(cci_scores.compute_jaccard_like_cci_score(cell, cell), 1.0)


def test_jaccard_like_formula():
    cell1 = FakeCell([1.0, 2.0], [0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0], [3.0, 4.0])
    c1 = np.array([1.0, 2.0])
    c2 = np.array([3.0, 4.0])
    numerator = np.sum(c1 * c2)
    denominator = np.sum(c1 * c1) + np.sum(c2 * c2) - numerator
    expected = numerator / denominator
    assert np.isclose(cci_scores.compute_jaccard_like_cci_score(cell1, cell2), expected)


def test_braycurtis_like_formula():
    cell1 = FakeCell([1.0, 2.0], [0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0], [3.0, 4.0])
    c1 = np.array([1.0, 2.0])
    c2 = np.array([3.0, 4.0])
    expected = 2 * np.sum(c1 * c2) / (np.sum(c1 * c1) + np.sum(c2 * c2))
    assert np.isclose(cci_scores.compute_braycurtis_like_cci_score(cell1, cell2), expected)


def test_count_score_counts_active_interactions():
    cell1 = FakeCell([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    # Only positions 0 and 2 are active in both
    assert cci_scores.compute_count_score(cell1, cell2) == 2.0


def test_count_score_is_zero_without_overlap():
    cell1 = FakeCell([1.0, 0.0], [0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0], [0.0, 1.0])
    assert cci_scores.compute_count_score(cell1, cell2) == 0.0


def test_icellnet_score_formula():
    cell1 = FakeCell([2.0, 3.0], [0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0], [4.0, 5.0])
    expected = np.nansum(np.array([2.0, 3.0]) * np.array([4.0, 5.0]))
    assert np.isclose(cci_scores.compute_icellnet_score(cell1, cell2), expected)


@pytest.mark.parametrize('score_function', PAIRWISE)
def test_ppi_score_weighting_changes_the_result(score_function):
    cell1 = FakeCell([1.0, 2.0], [0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0], [3.0, 4.0])
    unweighted = score_function(cell1, cell2)
    weighted = score_function(cell1, cell2, ppi_score=np.array([1.0, 0.0]))
    assert not np.isclose(unweighted, weighted)


def test_ppi_score_of_ones_matches_the_default():
    cell1 = FakeCell([1.0, 2.0], [0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0], [3.0, 4.0])
    default = cci_scores.compute_jaccard_like_cci_score(cell1, cell2)
    explicit = cci_scores.compute_jaccard_like_cci_score(cell1, cell2,
                                                        ppi_score=np.array([1.0, 1.0]))
    assert np.isclose(default, explicit)


# ---------------------------------------------------------------------------------
# Matrix implementations must agree with the pairwise ones
# ---------------------------------------------------------------------------------

@pytest.fixture
def score_matrices():
    '''A_scores and B_scores: rows are PPIs, columns are cells.'''
    a_scores = np.array([[1.0, 0.5, 0.0],
                         [0.2, 1.0, 0.7],
                         [0.9, 0.0, 0.3]])
    b_scores = np.array([[0.3, 0.9, 1.0],
                         [1.0, 0.1, 0.4],
                         [0.5, 0.6, 0.2]])
    return a_scores, b_scores


def test_matmul_jaccard_like_matches_pairwise(score_matrices):
    a_scores, b_scores = score_matrices
    matrix = cci_scores.matmul_jaccard_like(a_scores, b_scores)
    for i in range(a_scores.shape[1]):
        for j in range(b_scores.shape[1]):
            cell1 = FakeCell(a_scores[:, i], b_scores[:, i])
            cell2 = FakeCell(a_scores[:, j], b_scores[:, j])
            expected = cci_scores.compute_jaccard_like_cci_score(cell1, cell2)
            assert np.isclose(matrix[i, j], expected)


def test_matmul_bray_curtis_like_matches_pairwise(score_matrices):
    a_scores, b_scores = score_matrices
    matrix = cci_scores.matmul_bray_curtis_like(a_scores, b_scores)
    for i in range(a_scores.shape[1]):
        for j in range(b_scores.shape[1]):
            cell1 = FakeCell(a_scores[:, i], b_scores[:, i])
            cell2 = FakeCell(a_scores[:, j], b_scores[:, j])
            expected = cci_scores.compute_braycurtis_like_cci_score(cell1, cell2)
            assert np.isclose(matrix[i, j], expected)


def test_matmul_count_active_matches_pairwise():
    a_scores = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    b_scores = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    matrix = cci_scores.matmul_count_active(a_scores, b_scores)
    for i in range(2):
        for j in range(2):
            cell1 = FakeCell(a_scores[:, i], b_scores[:, i])
            cell2 = FakeCell(a_scores[:, j], b_scores[:, j])
            expected = cci_scores.compute_count_score(cell1, cell2)
            assert np.isclose(matrix[i, j], expected)


def test_matmul_shapes(score_matrices):
    a_scores, b_scores = score_matrices
    n_cells = a_scores.shape[1]
    for function in [cci_scores.matmul_jaccard_like, cci_scores.matmul_bray_curtis_like,
                     cci_scores.matmul_count_active, matmul_cosine]:
        assert function(a_scores, b_scores).shape == (n_cells, n_cells)


def test_matmul_cosine_is_bounded(score_matrices):
    a_scores, b_scores = score_matrices
    result = matmul_cosine(a_scores, b_scores)
    assert (result >= -1.0).all() and (result <= 1.0).all()


def test_matmul_accepts_a_ppi_score(score_matrices):
    a_scores, b_scores = score_matrices
    ppi_score = np.array([1.0, 0.5, 0.0])
    matrix = cci_scores.matmul_jaccard_like(a_scores, b_scores, ppi_score=ppi_score)
    for i in range(a_scores.shape[1]):
        for j in range(b_scores.shape[1]):
            cell1 = FakeCell(a_scores[:, i], b_scores[:, i])
            cell2 = FakeCell(a_scores[:, j], b_scores[:, j])
            expected = cci_scores.compute_jaccard_like_cci_score(cell1, cell2,
                                                                 ppi_score=ppi_score)
            assert np.isclose(matrix[i, j], expected)
