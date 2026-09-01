# -*- coding: utf-8 -*-

'''Tests for cell2cell.preprocessing.ppi'''

import numpy as np
import pandas as pd
import pytest

from cell2cell.preprocessing import ppi as ppi_module


COLUMNS = ('A', 'B')


# ---------------------------------------------------------------------------------
# remove_ppi_bidirectionality
# ---------------------------------------------------------------------------------

def test_remove_ppi_bidirectionality_keeps_one_direction():
    ppi = pd.DataFrame({'A': ['G1', 'G2', 'G3'], 'B': ['G2', 'G1', 'G4']})
    result = ppi_module.remove_ppi_bidirectionality(ppi, COLUMNS, verbose=False)
    pairs = set(zip(result['A'], result['B']))
    assert len(pairs & {('G1', 'G2'), ('G2', 'G1')}) == 1
    assert ('G3', 'G4') in pairs


def test_remove_ppi_bidirectionality_keeps_unidirectional_interactions(toy_ppi):
    result = ppi_module.remove_ppi_bidirectionality(toy_ppi, COLUMNS, verbose=False)
    assert result.shape[0] <= toy_ppi.shape[0]
    assert result.shape[0] > 0


def test_remove_ppi_bidirectionality_is_idempotent(toy_ppi):
    once = ppi_module.remove_ppi_bidirectionality(toy_ppi, COLUMNS, verbose=False)
    twice = ppi_module.remove_ppi_bidirectionality(once, COLUMNS, verbose=False)
    assert once.shape[0] == twice.shape[0]


def test_remove_ppi_bidirectionality_preserves_self_interactions():
    ppi = pd.DataFrame({'A': ['G1'], 'B': ['G1']})
    result = ppi_module.remove_ppi_bidirectionality(ppi, COLUMNS, verbose=False)
    assert result.shape[0] == 1


def test_remove_ppi_bidirectionality_keeps_a_one_way_interaction_between_paired_proteins():
    '''G4-G1 must survive: G1-G4 is not in the table, so it is not a duplicate.

    Both of its partners take part in a reciprocal interaction elsewhere, which used
    to be enough for it to be deleted.
    '''
    ppi = pd.DataFrame({'A': ['G1', 'G2', 'G3', 'G4', 'G4'],
                        'B': ['G2', 'G1', 'G4', 'G3', 'G1']})
    result = ppi_module.remove_ppi_bidirectionality(ppi, COLUMNS, verbose=False)
    pairs = list(zip(result['A'], result['B']))

    assert ('G4', 'G1') in pairs
    assert len([p for p in pairs if set(p) == {'G1', 'G2'}]) == 1
    assert len([p for p in pairs if set(p) == {'G3', 'G4'}]) == 1
    assert len(pairs) == 3


def test_remove_ppi_bidirectionality_keeps_the_lexicographic_orientation():
    '''Which orientation survives decides which partner is the ligand, so it is pinned.

    Independent of the order the two directions appear in, as it was before.
    '''
    for frame in (pd.DataFrame({'A': ['G2', 'G1'], 'B': ['G1', 'G2']}),
                  pd.DataFrame({'A': ['G1', 'G2'], 'B': ['G2', 'G1']})):
        result = ppi_module.remove_ppi_bidirectionality(frame, COLUMNS, verbose=False)
        assert list(zip(result['A'], result['B'])) == [('G1', 'G2')]


def test_remove_ppi_bidirectionality_keeps_repeated_rows():
    '''A pair listed twice in the same direction is not a bidirectional duplicate.'''
    ppi = pd.DataFrame({'A': ['G1', 'G1'], 'B': ['G2', 'G2']})
    result = ppi_module.remove_ppi_bidirectionality(ppi, COLUMNS, verbose=False)
    assert result.shape[0] == 2


def test_remove_ppi_bidirectionality_keeps_the_other_columns():
    ppi = pd.DataFrame({'A': ['G1', 'G2'], 'B': ['G2', 'G1'],
                        'score': [0.5, 0.9], 'function': ['adhesion', 'signalling']})
    result = ppi_module.remove_ppi_bidirectionality(ppi, COLUMNS, verbose=False)
    assert list(result.columns) == list(ppi.columns)
    assert result.loc[0, 'score'] == 0.5
    assert result.loc[0, 'function'] == 'adhesion'


def test_remove_ppi_bidirectionality_on_empty_input():
    ppi = pd.DataFrame({'A': [], 'B': []})
    result = ppi_module.remove_ppi_bidirectionality(ppi, COLUMNS, verbose=False)
    assert result.shape[0] == 0


# ---------------------------------------------------------------------------------
# deduplicate_ppi_pairs
# ---------------------------------------------------------------------------------

@pytest.fixture
def repeated_ppi():
    '''G1-G2 listed three times with different weights, plus one unrepeated pair.'''
    return pd.DataFrame({'A': ['G1', 'G3', 'G1', 'G1'],
                         'B': ['G2', 'G4', 'G2', 'G2'],
                         'score': [0.5, 1.0, 0.9, 0.2]})


def test_deduplicate_ppi_pairs_keeps_the_highest_score_by_default(repeated_ppi):
    result = ppi_module.deduplicate_ppi_pairs(repeated_ppi, COLUMNS)
    assert list(zip(result['A'], result['B'])) == [('G1', 'G2'), ('G3', 'G4')]
    assert result.loc[0, 'score'] == 0.9


def test_deduplicate_ppi_pairs_can_keep_the_lowest_score(repeated_ppi):
    result = ppi_module.deduplicate_ppi_pairs(repeated_ppi, COLUMNS, keep='lowest')
    assert result.loc[0, 'score'] == 0.2


def test_deduplicate_ppi_pairs_can_keep_the_first_row(repeated_ppi):
    result = ppi_module.deduplicate_ppi_pairs(repeated_ppi, COLUMNS, keep='first')
    assert result.loc[0, 'score'] == 0.5


def test_deduplicate_ppi_pairs_keeps_the_order_pairs_first_appear(repeated_ppi):
    '''Order matters: it is what the candidate sets of the search are indexed against.'''
    result = ppi_module.deduplicate_ppi_pairs(repeated_ppi, COLUMNS)
    assert list(zip(result['A'], result['B'])) == [('G1', 'G2'), ('G3', 'G4')]


def test_deduplicate_ppi_pairs_without_a_score_column():
    ppi = pd.DataFrame({'A': ['G1', 'G1', 'G3'], 'B': ['G2', 'G2', 'G4'],
                        'source': ['curated', 'predicted', 'curated']})
    result = ppi_module.deduplicate_ppi_pairs(ppi, COLUMNS)
    assert result.shape[0] == 2
    assert result.loc[0, 'source'] == 'curated'      # nothing to compare, so the first


def test_deduplicate_ppi_pairs_treats_a_reversed_pair_as_a_different_pair():
    '''Direction is meaningful for ligand-receptor pairs; reciprocals are a separate step.'''
    ppi = pd.DataFrame({'A': ['G1', 'G2'], 'B': ['G2', 'G1'], 'score': [0.5, 0.9]})
    result = ppi_module.deduplicate_ppi_pairs(ppi, COLUMNS)
    assert result.shape[0] == 2


def test_deduplicate_ppi_pairs_rejects_an_unknown_rule(repeated_ppi):
    with pytest.raises(ValueError):
        ppi_module.deduplicate_ppi_pairs(repeated_ppi, COLUMNS, keep='cheapest')


def test_preprocess_ppi_data_collapses_pairs_that_differ_only_in_score():
    '''One interaction listed twice would otherwise contribute twice to every score.'''
    ppi = pd.DataFrame({'A': ['G1', 'G1', 'G3'], 'B': ['G2', 'G2', 'G4'],
                        'weight': [0.5, 0.9, 1.0]})
    result = ppi_module.preprocess_ppi_data(ppi, COLUMNS, score='weight', verbose=False)
    assert result.shape[0] == 2
    assert result.loc[result['A'] == 'G1', 'score'].tolist() == [0.9]


def test_preprocess_ppi_data_can_keep_repeated_pairs():
    ppi = pd.DataFrame({'A': ['G1', 'G1', 'G3'], 'B': ['G2', 'G2', 'G4'],
                        'weight': [0.5, 0.9, 1.0]})
    result = ppi_module.preprocess_ppi_data(ppi, COLUMNS, score='weight',
                                            duplicates='keep', verbose=False)
    assert result.shape[0] == 3


# ---------------------------------------------------------------------------------
# simplify_ppi / preprocess_ppi_data
# ---------------------------------------------------------------------------------

def test_simplify_ppi_renames_to_abscore(toy_ppi):
    result = ppi_module.simplify_ppi(toy_ppi, COLUMNS, verbose=False)
    assert list(result.columns) == ['A', 'B', 'score']
    assert result.shape[0] == toy_ppi.shape[0]


def test_simplify_ppi_default_score_is_one(toy_ppi):
    result = ppi_module.simplify_ppi(toy_ppi, COLUMNS, verbose=False)
    assert (result['score'] == 1.0).all()


def test_simplify_ppi_uses_an_existing_column_as_the_score(toy_ppi):
    '''`score` names a column in ppi_data, it is not a constant value.'''
    weighted = toy_ppi.copy()
    weighted['weight'] = np.linspace(0.1, 1.0, weighted.shape[0])
    result = ppi_module.simplify_ppi(weighted, COLUMNS, score='weight', verbose=False)
    assert np.allclose(result['score'].values, weighted['weight'].values)


def test_simplify_ppi_fills_missing_scores_with_the_minimum(toy_ppi):
    weighted = toy_ppi.copy()
    weighted['weight'] = [0.4] * weighted.shape[0]
    weighted.loc[0, 'weight'] = np.nan
    result = ppi_module.simplify_ppi(weighted, COLUMNS, score='weight', verbose=False)
    assert np.isclose(result.loc[0, 'score'], 0.4)
    assert not result['score'].isna().any()


def test_preprocess_ppi_data_sorts_when_requested(toy_ppi):
    result = ppi_module.preprocess_ppi_data(toy_ppi, COLUMNS, sort_values='A',
                                            verbose=False)
    assert list(result['A']) == sorted(result['A'])


def test_preprocess_ppi_data_filters_by_genes(toy_ppi):
    genes = ['Protein-A', 'Protein-B']
    result = ppi_module.preprocess_ppi_data(toy_ppi, COLUMNS, rnaseq_genes=genes,
                                            verbose=False)
    # Names are upper-cased, since upper_letter_comparison defaults to True
    found = set(result['A']).union(result['B'])
    assert found.issubset({gene.upper() for gene in genes})


def test_preprocess_ppi_data_removes_duplicates():
    ppi = pd.DataFrame({'A': ['G1', 'G1'], 'B': ['G2', 'G2'], 'score': [1.0, 1.0]})
    result = ppi_module.preprocess_ppi_data(ppi, COLUMNS, verbose=False)
    assert result.shape[0] == 1


# ---------------------------------------------------------------------------------
# filter_ppi_by_proteins
# ---------------------------------------------------------------------------------

def test_filter_ppi_by_proteins_keeps_only_listed_proteins(toy_ppi):
    proteins = ['Protein-A', 'Protein-B']
    result = ppi_module.filter_ppi_by_proteins(toy_ppi, proteins,
                                              interaction_columns=COLUMNS)
    # upper_letter_comparison=True also upper-cases the names in the output
    found = set(result['A']).union(result['B'])
    assert found.issubset({p.upper() for p in proteins})


def test_filter_ppi_by_proteins_preserves_case_when_not_comparing_upper(toy_ppi):
    proteins = ['Protein-A', 'Protein-B']
    result = ppi_module.filter_ppi_by_proteins(toy_ppi, proteins,
                                              upper_letter_comparison=False,
                                              interaction_columns=COLUMNS)
    assert set(result['A']).union(result['B']).issubset(set(proteins))


def test_filter_ppi_by_proteins_is_case_insensitive_when_asked(toy_ppi):
    result = ppi_module.filter_ppi_by_proteins(toy_ppi, ['protein-a', 'protein-b'],
                                              upper_letter_comparison=True,
                                              interaction_columns=COLUMNS)
    assert result.shape[0] > 0


def test_filter_ppi_by_proteins_case_sensitive(toy_ppi):
    result = ppi_module.filter_ppi_by_proteins(toy_ppi, ['protein-a'],
                                              upper_letter_comparison=False,
                                              interaction_columns=COLUMNS)
    assert result.shape[0] == 0


def test_filter_ppi_by_proteins_with_complexes(toy_ppi_complex):
    result = ppi_module.filter_ppi_by_proteins(toy_ppi_complex,
                                               ['Protein-C', 'Protein-E', 'Protein-F'],
                                               complex_sep='&',
                                               interaction_columns=COLUMNS)
    assert result.shape[0] > 0


def test_filter_ppi_by_proteins_with_nothing_matching(toy_ppi):
    result = ppi_module.filter_ppi_by_proteins(toy_ppi, ['Not-A-Gene'],
                                               interaction_columns=COLUMNS)
    assert result.shape[0] == 0


# ---------------------------------------------------------------------------------
# Complexes
# ---------------------------------------------------------------------------------

def test_get_genes_from_complexes_returns_five_collections(toy_ppi_complex):
    '''Returns (col_a_genes, complex_a, col_b_genes, complex_b, complexes).'''
    result = ppi_module.get_genes_from_complexes(toy_ppi_complex, complex_sep='&',
                                                interaction_columns=COLUMNS)
    assert len(result) == 5
    col_a_genes, complex_a, col_b_genes, complex_b, complexes = result
    assert isinstance(complexes, dict)
    for name, subunits in complexes.items():
        assert '&' in name
        assert len(subunits) > 1
        assert set(name.split('&')) == set(subunits)
        for subunit in subunits:
            assert '&' not in subunit


def test_get_genes_from_complexes_separates_single_genes(toy_ppi_complex):
    col_a_genes, complex_a, col_b_genes, complex_b, _ = \
        ppi_module.get_genes_from_complexes(toy_ppi_complex, complex_sep='&',
                                           interaction_columns=COLUMNS)
    # Single-gene entries never contain the separator
    for gene in set(col_a_genes).union(col_b_genes):
        assert '&' not in gene
    # Subunits collected from the complexes are single genes too
    for gene in set(complex_a).union(complex_b):
        assert '&' not in gene


def test_get_genes_from_complexes_without_complexes(toy_ppi):
    _, complex_a, _, complex_b, complexes = \
        ppi_module.get_genes_from_complexes(toy_ppi, complex_sep='&',
                                           interaction_columns=COLUMNS)
    assert complexes == {}
    assert complex_a == set() and complex_b == set()


def test_filter_complex_ppi_by_proteins_requires_all_subunits(toy_ppi_complex):
    # 'Protein-C&Protein-E' needs both subunits present
    with_both = ppi_module.filter_complex_ppi_by_proteins(
        toy_ppi_complex, ['Protein-C', 'Protein-E', 'Protein-F'], complex_sep='&',
        interaction_columns=COLUMNS)
    assert any('&' in value for value in with_both['A'])


# ---------------------------------------------------------------------------------
# bidirectional_ppi_for_cci
# ---------------------------------------------------------------------------------

def test_bidirectional_ppi_for_cci_doubles_the_interactions():
    ppi = pd.DataFrame({'A': ['G1', 'G3'], 'B': ['G2', 'G4'], 'score': [1.0, 1.0]})
    result = ppi_module.bidirectional_ppi_for_cci(ppi, COLUMNS, verbose=False)
    pairs = set(zip(result['A'], result['B']))
    assert ('G1', 'G2') in pairs and ('G2', 'G1') in pairs
    assert ('G3', 'G4') in pairs and ('G4', 'G3') in pairs


def test_bidirectional_ppi_for_cci_does_not_duplicate_self_interactions():
    ppi = pd.DataFrame({'A': ['G1'], 'B': ['G1'], 'score': [1.0]})
    result = ppi_module.bidirectional_ppi_for_cci(ppi, COLUMNS, verbose=False)
    assert result.shape[0] == 1


def test_bidirectional_ppi_for_cci_on_empty_input():
    ppi = pd.DataFrame({'A': [], 'B': [], 'score': []})
    result = ppi_module.bidirectional_ppi_for_cci(ppi, COLUMNS, verbose=False)
    assert result.shape[0] == 0


# ---------------------------------------------------------------------------------
# get_all_to_all_ppi / get_one_group_to_other_ppi
# ---------------------------------------------------------------------------------

def test_get_all_to_all_ppi_needs_both_sides_listed(toy_ppi):
    result = ppi_module.get_all_to_all_ppi(toy_ppi, ['Protein-A', 'Protein-B'],
                                           interaction_columns=COLUMNS)
    for _, row in result.iterrows():
        assert row['A'] in ['Protein-A', 'Protein-B']
        assert row['B'] in ['Protein-A', 'Protein-B']


def test_get_one_group_to_other_ppi_is_directional(toy_ppi):
    result = ppi_module.get_one_group_to_other_ppi(toy_ppi, proteins_a=['Protein-A'],
                                                   proteins_b=['Protein-B'],
                                                   interaction_columns=COLUMNS)
    for _, row in result.iterrows():
        assert row['A'] == 'Protein-A'
        assert row['B'] == 'Protein-B'


# ---------------------------------------------------------------------------------
# filter_ppi_network / get_filtered_ppi_network
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize('interaction_type', ['contacts', 'complete'])
def test_get_filtered_ppi_network_types(toy_ppi, interaction_type):
    result = ppi_module.get_filtered_ppi_network(
        ppi_data=toy_ppi,
        contact_proteins=['Protein-A', 'Protein-B'],
        mediator_proteins=['Protein-E', 'Protein-F'],
        interaction_type=interaction_type,
        interaction_columns=COLUMNS,
        verbose=False)
    assert list(result.columns) == ['A', 'B', 'score']


def test_get_filtered_ppi_network_is_reproducible(toy_ppi):
    kwargs = dict(ppi_data=toy_ppi, contact_proteins=['Protein-B', 'Protein-A'],
                  mediator_proteins=['Protein-F', 'Protein-E'],
                  interaction_type='complete', interaction_columns=COLUMNS,
                  verbose=False)
    first = ppi_module.get_filtered_ppi_network(**kwargs)
    second = ppi_module.get_filtered_ppi_network(**kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_filter_ppi_network_returns_a_dataframe(toy_ppi):
    result = ppi_module.filter_ppi_network(
        ppi_data=toy_ppi,
        contact_proteins=['Protein-A', 'Protein-B'],
        mediator_proteins=['Protein-E', 'Protein-F'],
        interaction_type='combined',
        interaction_columns=COLUMNS,
        verbose=False)
    assert isinstance(result, pd.DataFrame)


def test_ppi_functions_do_not_modify_input(toy_ppi):
    before = toy_ppi.copy()
    ppi_module.remove_ppi_bidirectionality(toy_ppi, COLUMNS, verbose=False)
    ppi_module.simplify_ppi(toy_ppi, COLUMNS, verbose=False)
    ppi_module.filter_ppi_by_proteins(toy_ppi, ['Protein-A'], interaction_columns=COLUMNS)
    ppi_module.bidirectional_ppi_for_cci(toy_ppi, COLUMNS, verbose=False)
    pd.testing.assert_frame_equal(toy_ppi, before)


# ---------------------------------------------------------------------------------
# Deliberate behaviour -- guards against a future "fix" that would break it
# ---------------------------------------------------------------------------------

def test_remove_ppi_bidirectionality_keeps_using_lexicographic_order():
    '''This lexicographic sort decides WHICH direction of a bidirectional PPI is
    dropped. Replacing it with a natural sort would silently change which rows
    survive, so the output is pinned here.
    '''
    ppi = pd.DataFrame({'A': ['G1', 'G2', 'G3', 'G2', 'G10', 'G2'],
                        'B': ['G2', 'G1', 'G4', 'G3', 'G2', 'G10']})
    result = ppi_module.remove_ppi_bidirectionality(ppi, ('A', 'B'), verbose=False)

    pairs = set(zip(result['A'], result['B']))
    # Of each bidirectional pair only one direction is kept
    assert ('G1', 'G2') in pairs and ('G2', 'G1') not in pairs
    assert ('G10', 'G2') in pairs and ('G2', 'G10') not in pairs
    # Unidirectional interactions are untouched
    assert ('G3', 'G4') in pairs
    assert result.shape[0] == 4
