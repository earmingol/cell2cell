# -*- coding: utf-8 -*-

'''Tests for cell2cell.preprocessing.integrate_data'''

import numpy as np
import pandas as pd
import pytest

from cell2cell.preprocessing import cutoffs as cutoffs_module
from cell2cell.preprocessing import integrate_data


@pytest.fixture
def constant_cutoffs(toy_rnaseq):
    return cutoffs_module.get_constant_cutoff(toy_rnaseq, constant_cutoff=10)


# ---------------------------------------------------------------------------------
# get_thresholded_rnaseq
# ---------------------------------------------------------------------------------

def test_get_thresholded_rnaseq_is_binary(toy_rnaseq, constant_cutoffs):
    result = integrate_data.get_thresholded_rnaseq(toy_rnaseq, constant_cutoffs)
    assert set(np.unique(result.values)).issubset({0.0, 1.0})
    assert result.shape == toy_rnaseq.shape


def test_get_thresholded_rnaseq_uses_strictly_greater_than(toy_rnaseq, constant_cutoffs):
    result = integrate_data.get_thresholded_rnaseq(toy_rnaseq, constant_cutoffs)
    expected = (toy_rnaseq > 10).astype(float)
    pd.testing.assert_frame_equal(result, expected)


def test_get_thresholded_rnaseq_accepts_per_cell_cutoffs(toy_rnaseq):
    per_cell = pd.DataFrame(10.0, index=toy_rnaseq.index, columns=toy_rnaseq.columns)
    per_cell['C1'] = 0.0            # everything in C1 passes
    result = integrate_data.get_thresholded_rnaseq(toy_rnaseq, per_cell)
    assert (result['C1'] == 1.0).all()


def test_get_thresholded_rnaseq_rejects_mismatched_cutoffs(toy_rnaseq):
    bad = pd.DataFrame({'unexpected': [1.0] * toy_rnaseq.shape[0]}, index=toy_rnaseq.index)
    with pytest.raises(KeyError):
        integrate_data.get_thresholded_rnaseq(toy_rnaseq, bad)


def test_get_thresholded_rnaseq_does_not_modify_input(toy_rnaseq, constant_cutoffs):
    before = toy_rnaseq.copy()
    integrate_data.get_thresholded_rnaseq(toy_rnaseq, constant_cutoffs)
    pd.testing.assert_frame_equal(toy_rnaseq, before)


# ---------------------------------------------------------------------------------
# get_modified_rnaseq
# ---------------------------------------------------------------------------------

def test_get_modified_rnaseq_thresholding(toy_rnaseq, constant_cutoffs):
    result = integrate_data.get_modified_rnaseq(toy_rnaseq, cutoffs=constant_cutoffs,
                                                communication_score='expression_thresholding')
    expected = integrate_data.get_thresholded_rnaseq(toy_rnaseq, constant_cutoffs)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize('score', ['expression_product', 'expression_mean',
                                   'expression_gmean'])
def test_get_modified_rnaseq_continuous_scores_are_a_copy(toy_rnaseq, score):
    result = integrate_data.get_modified_rnaseq(toy_rnaseq, communication_score=score)
    pd.testing.assert_frame_equal(result, toy_rnaseq)
    assert result is not toy_rnaseq


def test_get_modified_rnaseq_rejects_unknown_score(toy_rnaseq):
    with pytest.raises(NotImplementedError):
        integrate_data.get_modified_rnaseq(toy_rnaseq, communication_score='nonsense')


# ---------------------------------------------------------------------------------
# get_weighted_ppi
# ---------------------------------------------------------------------------------

def test_get_weighted_ppi_replaces_genes_with_expression(toy_rnaseq, toy_ppi):
    modified = toy_rnaseq[['C1']].rename(columns={'C1': 'value'})
    result = integrate_data.get_weighted_ppi(toy_ppi, modified, column='value',
                                             interaction_columns=('A', 'B'))
    assert list(result.columns) == ['A', 'B', 'score']
    assert result.shape[0] == toy_ppi.shape[0]
    for i, row in toy_ppi.iterrows():
        assert np.isclose(result.loc[i, 'A'], modified.at[row['A'], 'value'])
        assert np.isclose(result.loc[i, 'B'], modified.at[row['B'], 'value'])


def test_get_weighted_ppi_does_not_modify_input(toy_rnaseq, toy_ppi):
    modified = toy_rnaseq[['C1']].rename(columns={'C1': 'value'})
    before = toy_ppi.copy()
    integrate_data.get_weighted_ppi(toy_ppi, modified)
    pd.testing.assert_frame_equal(toy_ppi, before)


# ---------------------------------------------------------------------------------
# get_ppi_dict_from_proteins
# ---------------------------------------------------------------------------------

def test_get_ppi_dict_from_proteins_contacts_only(toy_ppi):
    result = integrate_data.get_ppi_dict_from_proteins(
        toy_ppi, contact_proteins=['Protein-A', 'Protein-B'],
        interaction_columns=('A', 'B'), verbose=False)
    assert 'contacts' in result
    for df in result.values():
        assert list(df.columns)[:2] == ['A', 'B']


def test_get_ppi_dict_from_proteins_with_mediators(toy_ppi):
    result = integrate_data.get_ppi_dict_from_proteins(
        toy_ppi, contact_proteins=['Protein-A', 'Protein-B'],
        mediator_proteins=['Protein-E', 'Protein-F'],
        interaction_columns=('A', 'B'), verbose=False)
    assert 'contacts' in result and 'mediated' in result
    assert 'combined' in result


def test_get_ppi_dict_from_proteins_is_reproducible(toy_ppi):
    kwargs = dict(contact_proteins=['Protein-B', 'Protein-A'],
                  mediator_proteins=['Protein-F', 'Protein-E'],
                  interaction_columns=('A', 'B'), verbose=False)
    first = integrate_data.get_ppi_dict_from_proteins(toy_ppi, **kwargs)
    second = integrate_data.get_ppi_dict_from_proteins(toy_ppi, **kwargs)
    assert list(first.keys()) == list(second.keys())
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])


# ---------------------------------------------------------------------------------
# get_ppi_dict_from_go_terms
# ---------------------------------------------------------------------------------

def test_get_ppi_dict_from_go_terms(toy_ppi, go_terms_graph, go_annotations):
    result = integrate_data.get_ppi_dict_from_go_terms(
        ppi_data=toy_ppi,
        go_annotations=go_annotations,
        go_terms=go_terms_graph,
        contact_go_terms=['GO:0000002'],
        mediator_go_terms=['GO:0000003'],
        go_header='go_id',
        gene_header='db_object_symbol',
        interaction_columns=('A', 'B'),
        verbose=False)
    assert 'contacts' in result and 'mediated' in result
