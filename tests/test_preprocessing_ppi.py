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


# ---------------------------------------------------------------------------------
# filter_ppi_by_adata
#
# Cutting a ligand-receptor list down to the genes a panel measured, so that a pair
# the assay could not see is not scored as one the cells did not use.
# ---------------------------------------------------------------------------------

@pytest.fixture
def panel_ppi():
    '''A small LR list with complexes, and the genes a panel would have measured.'''
    ppi = pd.DataFrame({'ligand': ['LA', 'LB', 'LC&LD', 'LE&LF', 'LG', 'LH&LI'],
                        'receptor': ['RA', 'RB&RC', 'RD', 'RE', 'RF&RG', 'RH'],
                        'score': [1., 1., 1., 1., 1., 1.]})
    # LF, LI, RC, RF and RG were not measured; everything else was
    measured = ['LA', 'LB', 'LC', 'LD', 'LE', 'LG', 'LH',
                'RA', 'RB', 'RD', 'RE', 'RH']
    return ppi, measured


def test_filter_ppi_by_adata_keeps_only_measured_genes(panel_ppi):
    ppi, measured = panel_ppi
    kept, report = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        verbose=False)
    for column in ('ligand_filtered', 'receptor_filtered'):
        for partner in kept[column]:
            assert all(sub in measured for sub in partner.split('&'))


def test_filter_ppi_by_adata_trims_a_partly_measured_complex(panel_ppi):
    '''The default keeps the measured subunits rather than dropping the pair.'''
    ppi, measured = panel_ppi
    kept, _ = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        verbose=False)
    row = kept[kept['ligand'] == 'LE&LF'].iloc[0]
    assert row['ligand_filtered'] == 'LE'        # LF was not measured
    assert row['ligand'] == 'LE&LF'              # the original is left alone


def test_filter_ppi_by_adata_strict_drops_a_partly_measured_complex(panel_ppi):
    ppi, measured = panel_ppi
    kept, report = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        complex_policy='strict', verbose=False)
    assert 'LE&LF' not in set(kept['ligand'])
    assert 'LC&LD' in set(kept['ligand'])        # both subunits measured, so it stays
    assert report.loc['interactions', 'trimmed'] == 0


def test_filter_ppi_by_adata_strict_keeps_fewer_than_trim(panel_ppi):
    ppi, measured = panel_ppi
    trimmed, _ = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        complex_policy='trim', verbose=False)
    strict, _ = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        complex_policy='strict', verbose=False)
    assert len(strict) < len(trimmed)


def test_filter_ppi_by_adata_drops_a_pair_with_nothing_left(panel_ppi):
    '''RF&RG has no measured subunit, so that interaction cannot be scored at all.'''
    ppi, measured = panel_ppi
    kept, _ = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        verbose=False)
    assert 'LG' not in set(kept['ligand'])


def test_filter_ppi_by_adata_reports_what_it_cost(panel_ppi):
    ppi, measured = panel_ppi
    _, report = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        verbose=False)
    assert list(report.index) == ['ligand genes', 'receptor genes', 'interactions']

    # 9 distinct ligand subunits, of which LF and LI were not measured
    assert report.loc['ligand genes', 'total'] == 9
    assert report.loc['ligand genes', 'dropped'] == 2
    assert np.isclose(report.loc['ligand genes', 'fraction_dropped'], 2 / 9)

    # 8 distinct receptor subunits, of which RC, RF and RG were not measured
    assert report.loc['receptor genes', 'total'] == 8
    assert report.loc['receptor genes', 'dropped'] == 3

    assert report.loc['interactions', 'total'] == len(ppi)
    assert (report['kept'] + report['dropped'] == report['total']).all()


def test_filter_ppi_by_adata_counts_trimmed_interactions(panel_ppi):
    ppi, measured = panel_ppi
    kept, report = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        verbose=False)
    shortened = ((kept['ligand_filtered'] != kept['ligand'])
                 | (kept['receptor_filtered'] != kept['receptor'])).sum()
    assert report.loc['interactions', 'trimmed'] == shortened
    assert shortened > 0


def test_filter_ppi_by_adata_without_a_complex_separator(panel_ppi):
    '''With complex_sep=None a partner is one gene, separator or not.'''
    ppi, measured = panel_ppi
    kept, report = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), verbose=False)
    assert set(kept['ligand']) == {'LA'}          # the only row with both partners measured
    assert report.loc['ligand genes', 'total'] == len(ppi)


def test_filter_ppi_by_adata_accepts_an_anndata(panel_ppi):
    '''The genes are read off `var_names`, which is the point of passing an AnnData.'''
    anndata = pytest.importorskip('anndata')
    ppi, measured = panel_ppi
    adata = anndata.AnnData(np.zeros((3, len(measured))),
                            var=pd.DataFrame(index=measured))
    from_adata, _ = ppi_module.filter_ppi_by_adata(
        ppi, adata, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        verbose=False)
    from_list, _ = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        verbose=False)
    pd.testing.assert_frame_equal(from_adata, from_list)


def test_filter_ppi_by_adata_matches_case_insensitively_but_keeps_the_original(panel_ppi):
    ppi, measured = panel_ppi
    kept, _ = ppi_module.filter_ppi_by_adata(
        ppi, [g.lower() for g in measured], interaction_columns=('ligand', 'receptor'),
        complex_sep='&', verbose=False)
    assert len(kept) > 0
    # Matching ignored the case; the names written out kept the case they had
    subunits = {sub for partner in kept['ligand_filtered']
                for sub in partner.split('&')}
    assert subunits.issubset(set(measured))
    assert all(sub.isupper() for sub in subunits)


def test_filter_ppi_by_adata_can_be_case_sensitive(panel_ppi):
    ppi, measured = panel_ppi
    kept, _ = ppi_module.filter_ppi_by_adata(
        ppi, [g.lower() for g in measured], interaction_columns=('ligand', 'receptor'),
        complex_sep='&', upper_letter_comparison=False, verbose=False)
    assert len(kept) == 0


def test_filter_ppi_by_adata_names_its_new_columns(panel_ppi):
    ppi, measured = panel_ppi
    kept, _ = ppi_module.filter_ppi_by_adata(
        ppi, measured, interaction_columns=('ligand', 'receptor'), complex_sep='&',
        new_columns=('L', 'R'), verbose=False)
    assert 'L' in kept.columns and 'R' in kept.columns
    assert 'ligand' in kept.columns and 'score' in kept.columns


def test_filter_ppi_by_adata_when_nothing_was_measured(panel_ppi):
    ppi, _ = panel_ppi
    kept, report = ppi_module.filter_ppi_by_adata(
        ppi, [], interaction_columns=('ligand', 'receptor'), complex_sep='&',
        verbose=False)
    assert len(kept) == 0
    assert report.loc['interactions', 'dropped'] == len(ppi)
    assert report.loc['interactions', 'trimmed'] == 0


def test_filter_ppi_by_adata_validates_its_arguments(panel_ppi):
    ppi, measured = panel_ppi
    with pytest.raises(ValueError, match='complex_policy'):
        ppi_module.filter_ppi_by_adata(ppi, measured,
                                       interaction_columns=('ligand', 'receptor'),
                                       complex_policy='nonsense')
    with pytest.raises(KeyError):
        ppi_module.filter_ppi_by_adata(ppi, measured,
                                       interaction_columns=('nope', 'receptor'))


def test_filter_ppi_by_adata_prints_a_summary(panel_ppi, capsys):
    ppi, measured = panel_ppi
    ppi_module.filter_ppi_by_adata(ppi, measured,
                                   interaction_columns=('ligand', 'receptor'),
                                   complex_sep='&', verbose=True)
    printed = capsys.readouterr().out
    assert 'ligand genes' in printed and 'interactions' in printed
    assert 'trimmed' in printed
