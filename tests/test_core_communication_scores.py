# -*- coding: utf-8 -*-

'''Tests for cell2cell.core.communication_scores'''

import numpy as np
import pandas as pd
import pytest

from cell2cell.core import communication_scores


class FakeCell:
    def __init__(self, a_values, b_values):
        self.weighted_ppi = pd.DataFrame({'A': a_values, 'B': b_values})


def test_get_binary_scores_is_the_product_of_indicators():
    cell1 = FakeCell([1.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0, 0.0], [1.0, 0.0, 1.0])
    result = communication_scores.get_binary_scores(cell1, cell2)
    assert np.allclose(result, [1.0, 0.0, 0.0])


def test_get_binary_scores_with_ppi_score():
    cell1 = FakeCell([1.0, 1.0], [0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0], [1.0, 1.0])
    result = communication_scores.get_binary_scores(cell1, cell2,
                                                    ppi_score=np.array([2.0, 0.5]))
    assert np.allclose(result, [2.0, 0.5])


def test_get_continuous_scores_expression_product():
    cell1 = FakeCell([2.0, 3.0], [0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0], [4.0, 5.0])
    result = communication_scores.get_continuous_scores(cell1, cell2,
                                                        method='expression_product')
    assert np.allclose(result, [8.0, 15.0])


def test_get_continuous_scores_expression_mean():
    cell1 = FakeCell([2.0, 3.0], [0.0, 0.0])
    cell2 = FakeCell([0.0, 0.0], [4.0, 5.0])
    result = communication_scores.get_continuous_scores(cell1, cell2,
                                                        method='expression_mean')
    assert np.allclose(result, [3.0, 4.0])


def test_get_continuous_scores_rejects_unknown_method():
    cell = FakeCell([1.0], [1.0])
    with pytest.raises(ValueError):
        communication_scores.get_continuous_scores(cell, cell, method='nonsense')


def test_score_expression_product_and_mean_are_elementwise():
    c1 = np.array([1.0, 2.0, 3.0])
    c2 = np.array([4.0, 5.0, 6.0])
    assert np.allclose(communication_scores.score_expression_product(c1, c2), c1 * c2)
    assert np.allclose(communication_scores.score_expression_mean(c1, c2), (c1 + c2) / 2.)


# ---------------------------------------------------------------------------------
# compute_ccc_matrix
# ---------------------------------------------------------------------------------

@pytest.fixture
def expression_vectors():
    prot_a = np.array([1.0, 2.0, 3.0])       # senders
    prot_b = np.array([4.0, 5.0])            # receivers
    return prot_a, prot_b


def test_compute_ccc_matrix_expression_product(expression_vectors):
    prot_a, prot_b = expression_vectors
    result = communication_scores.compute_ccc_matrix(prot_a, prot_b,
                                                     communication_score='expression_product')
    assert result.shape == (3, 2)
    assert np.allclose(result, np.outer(prot_a, prot_b))


def test_compute_ccc_matrix_expression_mean(expression_vectors):
    prot_a, prot_b = expression_vectors
    result = communication_scores.compute_ccc_matrix(prot_a, prot_b,
                                                     communication_score='expression_mean')
    expected = (np.outer(prot_a, np.ones(2)) + np.outer(np.ones(3), prot_b)) / 2.
    assert np.allclose(result, expected)


def test_compute_ccc_matrix_expression_gmean(expression_vectors):
    prot_a, prot_b = expression_vectors
    result = communication_scores.compute_ccc_matrix(prot_a, prot_b,
                                                     communication_score='expression_gmean')
    assert np.allclose(result, np.sqrt(np.outer(prot_a, prot_b)))


def test_compute_ccc_matrix_rejects_unknown_score(expression_vectors):
    prot_a, prot_b = expression_vectors
    with pytest.raises(ValueError):
        communication_scores.compute_ccc_matrix(prot_a, prot_b,
                                                communication_score='nonsense')


def test_compute_ccc_matrix_orientation_is_senders_by_receivers():
    '''Rows must correspond to the first vector and columns to the second.'''
    prot_a = np.array([0.0, 1.0])
    prot_b = np.array([1.0, 1.0, 1.0])
    result = communication_scores.compute_ccc_matrix(prot_a, prot_b,
                                                     communication_score='expression_product')
    assert result.shape == (2, 3)
    assert np.allclose(result[0, :], 0.0)
    assert np.allclose(result[1, :], 1.0)


# ---------------------------------------------------------------------------------
# aggregate_ccc_matrices
# ---------------------------------------------------------------------------------

@pytest.fixture
def ccc_matrices():
    return [np.array([[1.0, 4.0], [9.0, 16.0]]),
            np.array([[1.0, 1.0], [1.0, 1.0]])]


def test_aggregate_ccc_matrices_gmean(ccc_matrices):
    result = communication_scores.aggregate_ccc_matrices(ccc_matrices, method='gmean')
    expected = np.sqrt(ccc_matrices[0] * ccc_matrices[1])
    assert np.allclose(result, expected)


def test_aggregate_ccc_matrices_sum(ccc_matrices):
    result = communication_scores.aggregate_ccc_matrices(ccc_matrices, method='sum')
    assert np.allclose(result, ccc_matrices[0] + ccc_matrices[1])


def test_aggregate_ccc_matrices_mean(ccc_matrices):
    result = communication_scores.aggregate_ccc_matrices(ccc_matrices, method='mean')
    assert np.allclose(result, (ccc_matrices[0] + ccc_matrices[1]) / 2.)


def test_aggregate_ccc_matrices_rejects_unknown_method(ccc_matrices):
    with pytest.raises(ValueError):
        communication_scores.aggregate_ccc_matrices(ccc_matrices, method='nonsense')


def test_aggregate_ccc_matrices_preserves_shape(ccc_matrices):
    for method in ['gmean', 'sum', 'mean']:
        result = communication_scores.aggregate_ccc_matrices(ccc_matrices, method=method)
        assert result.shape == ccc_matrices[0].shape


def test_aggregate_a_single_matrix_is_a_no_op(ccc_matrices):
    result = communication_scores.aggregate_ccc_matrices([ccc_matrices[0]], method='mean')
    assert np.allclose(result, ccc_matrices[0])
