# -*- coding: utf-8 -*-

'''Tests for cell2cell.core.interaction_space'''

import itertools

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.core import interaction_space as ispace


# ---------------------------------------------------------------------------------
# generate_pairs
# ---------------------------------------------------------------------------------

def test_generate_pairs_directed_with_self_interaction():
    pairs = ispace.generate_pairs(['A', 'B'], 'directed', self_interaction=True)
    assert set(pairs) == {('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')}


def test_generate_pairs_directed_without_self_interaction():
    pairs = ispace.generate_pairs(['A', 'B'], 'directed', self_interaction=False)
    assert set(pairs) == {('A', 'B'), ('B', 'A')}


def test_generate_pairs_undirected_with_self_interaction():
    pairs = ispace.generate_pairs(['A', 'B', 'C'], 'undirected', self_interaction=True)
    assert len(pairs) == 6
    for a, b in pairs:
        assert (b, a) not in pairs or a == b


def test_generate_pairs_undirected_without_self_interaction():
    pairs = ispace.generate_pairs(['A', 'B', 'C'], 'undirected', self_interaction=False)
    assert set(pairs) == {('A', 'B'), ('A', 'C'), ('B', 'C')}


def test_generate_pairs_rejects_unknown_cci_type():
    with pytest.raises(NotImplementedError):
        ispace.generate_pairs(['A', 'B'], 'nonsense')


def test_generate_pairs_counts_for_directed(toy_rnaseq):
    cells = list(toy_rnaseq.columns)
    pairs = ispace.generate_pairs(cells, 'directed')
    assert len(pairs) == len(cells) ** 2


# ---------------------------------------------------------------------------------
# InteractionSpace
# ---------------------------------------------------------------------------------

def test_interaction_space_has_the_expected_elements(interaction_space):
    elements = interaction_space.interaction_elements
    for key in ['cell_names', 'pairs', 'cci_matrix', 'communication_matrix']:
        assert key in elements


def test_cci_matrix_is_square_and_labelled(interaction_space, toy_rnaseq):
    cci = interaction_space.interaction_elements['cci_matrix']
    assert list(cci.index) == list(cci.columns)
    assert set(cci.index) == set(toy_rnaseq.columns)


def test_cci_matrix_is_symmetric_for_undirected(interaction_space):
    cci = interaction_space.interaction_elements['cci_matrix']
    assert np.allclose(cci.values, cci.values.T)


def test_cci_matrix_diagonal_holds_autocrine_scores(interaction_space):
    '''The diagonal is not 1: it compares a cell's ligand profile against its own
    receptor profile (an autocrine interaction), not a vector with itself.
    '''
    cci = interaction_space.interaction_elements['cci_matrix']
    diagonal = np.diag(cci.values)
    assert ((diagonal >= 0) & (diagonal <= 1)).all()
    assert (diagonal > 0).any()


def test_cci_matrix_values_are_bounded_for_bray_curtis(interaction_space):
    cci = interaction_space.interaction_elements['cci_matrix']
    assert ((cci.values >= 0) & (cci.values <= 1)).all()


def test_distance_matrix_is_a_valid_distance(interaction_space):
    distance = interaction_space.distance_matrix
    assert np.allclose(np.diag(distance.values), 0.0)
    assert (distance.values >= 0).all()
    assert np.allclose(distance.values, distance.values.T)


def test_distance_matrix_keeps_the_cell_labels(interaction_space):
    '''The diagonal is zeroed by rebuilding the frame, so the labels must be carried over
    and stay aligned with the CCI matrix they are derived from.'''
    distance = interaction_space.distance_matrix
    cci = interaction_space.interaction_elements['cci_matrix']
    assert list(distance.index) == list(cci.index)
    assert list(distance.columns) == list(cci.columns)
    # Off-diagonal entries are the plain complement of the bray_curtis scores
    off_diagonal = ~np.eye(cci.shape[0], dtype=bool)
    assert np.allclose(distance.values[off_diagonal], 1 - cci.values[off_diagonal])


def test_communication_matrix_columns_use_the_semicolon_separator(interaction_space):
    communication = interaction_space.interaction_elements['communication_matrix']
    for column in communication.columns:
        assert ';' in column


def test_communication_matrix_columns_match_the_pairs(interaction_space):
    elements = interaction_space.interaction_elements
    expected = ['{};{}'.format(a, b) for a, b in elements['pairs']]
    assert list(elements['communication_matrix'].columns) == expected


def test_communication_matrix_values_are_finite(interaction_space):
    communication = interaction_space.interaction_elements['communication_matrix']
    assert np.isfinite(communication.values.astype(float)).all()


def test_pairwise_cci_scores_are_reproducible(toy_rnaseq, toy_ppi, analysis_setup,
                                              cutoff_setup):
    import cell2cell as c2c

    def build():
        space = c2c.analysis.initialize_interaction_space(
            rnaseq_data=toy_rnaseq, ppi_data=toy_ppi, cutoff_setup=cutoff_setup,
            analysis_setup=analysis_setup, complex_sep=None, verbose=False)
        space.compute_pairwise_cci_scores(verbose=False)
        return space.interaction_elements['cci_matrix']

    pd.testing.assert_frame_equal(build(), build())


def test_generate_interaction_elements_directed_pair_count(toy_rnaseq, toy_ppi):
    from cell2cell.preprocessing import integrate_data
    modified = integrate_data.get_modified_rnaseq(
        toy_rnaseq, communication_score='expression_product')
    elements = ispace.generate_interaction_elements(modified, toy_ppi,
                                                   cci_type='directed', verbose=False)
    n_cells = toy_rnaseq.shape[1]
    assert len(elements['pairs']) == n_cells ** 2
    assert len(elements['cell_names']) == n_cells


def test_interaction_space_excluded_cells(toy_rnaseq, toy_ppi, analysis_setup,
                                          cutoff_setup):
    import cell2cell as c2c
    space = c2c.analysis.initialize_interaction_space(
        rnaseq_data=toy_rnaseq, ppi_data=toy_ppi, cutoff_setup=cutoff_setup,
        analysis_setup=analysis_setup, excluded_cells=['C1'], complex_sep=None,
        verbose=False)
    assert 'C1' not in space.interaction_elements['cell_names']


# ---------------------------------------------------------------------------------
# Distance matrix for the 'count' and 'icellnet' CCI scores
#
# The branch was guarded by `if ~(cci_score in [...])`. Bitwise inversion of a bool
# gives -2/-1, which are both truthy, so the regularized-distance branch was dead
# code and those scores produced NEGATIVE distances (down to -7 on the toy data).
# ---------------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------------
# Reproducibility -- these orders came from an unsorted set() and varied per run
# ---------------------------------------------------------------------------------

def test_generate_pairs_follows_the_order_of_the_cells():
    pairs = ispace.generate_pairs(['C3', 'C1', 'C2'], 'directed')
    assert pairs == [('C3', 'C3'), ('C3', 'C1'), ('C3', 'C2'),
                     ('C1', 'C3'), ('C1', 'C1'), ('C1', 'C2'),
                     ('C2', 'C3'), ('C2', 'C1'), ('C2', 'C2')]


def test_generate_pairs_deduplicates_without_losing_order():
    pairs = ispace.generate_pairs(['A', 'A', 'B'], 'directed')
    assert pairs == [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]


def test_generate_pairs_is_reproducible():
    first = ispace.generate_pairs(['C3', 'C1', 'C2'], 'undirected')
    second = ispace.generate_pairs(['C3', 'C1', 'C2'], 'undirected')
    assert first == second
    assert len(first) == len(set(first))
