# -*- coding: utf-8 -*-

'''Tests for cell2cell.core.interaction_space'''

import itertools

import numpy as np
import pandas as pd
import pytest

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
