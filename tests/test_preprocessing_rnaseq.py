# -*- coding: utf-8 -*-

'''Tests for cell2cell.preprocessing.rnaseq'''

import numpy as np
import pandas as pd
import pytest

from cell2cell.preprocessing import rnaseq
from cell2cell.preprocessing.rnaseq import _trimean


# ---------------------------------------------------------------------------------
# drop_empty_genes
# ---------------------------------------------------------------------------------

def test_drop_empty_genes_removes_all_zero_rows(toy_rnaseq):
    data = toy_rnaseq.copy().astype(float)
    data.loc['Protein-A'] = 0.0
    result = rnaseq.drop_empty_genes(data)
    assert 'Protein-A' not in result.index
    assert result.shape[0] == toy_rnaseq.shape[0] - 1


def test_drop_empty_genes_removes_all_nan_rows(toy_rnaseq):
    data = toy_rnaseq.copy().astype(float)
    data.loc['Protein-B'] = np.nan
    result = rnaseq.drop_empty_genes(data)
    assert 'Protein-B' not in result.index


def test_drop_empty_genes_fills_remaining_nans(toy_rnaseq):
    data = toy_rnaseq.copy().astype(float)
    data.loc['Protein-C', 'C1'] = np.nan
    result = rnaseq.drop_empty_genes(data)
    assert 'Protein-C' in result.index
    assert result.loc['Protein-C', 'C1'] == 0.0
    assert not result.isna().any().any()


def test_drop_empty_genes_keeps_a_full_dataset(toy_rnaseq):
    result = rnaseq.drop_empty_genes(toy_rnaseq)
    assert result.shape == toy_rnaseq.shape


def test_drop_empty_genes_does_not_modify_input(toy_rnaseq):
    data = toy_rnaseq.copy().astype(float)
    data.loc['Protein-A'] = 0.0
    before = data.values.copy()
    rnaseq.drop_empty_genes(data)
    assert np.array_equal(data.values, before)


# ---------------------------------------------------------------------------------
# log10_transformation
# ---------------------------------------------------------------------------------

def test_log10_transformation_values(toy_rnaseq):
    result = rnaseq.log10_transformation(toy_rnaseq, addition=1e-6)
    expected = np.log10(toy_rnaseq.values + 1e-6)
    assert np.allclose(result.values, expected)
    assert list(result.index) == list(toy_rnaseq.index)


def test_log10_transformation_turns_infinities_into_nan():
    data = pd.DataFrame({'C1': [0.0, 1.0]}, index=['g1', 'g2'])
    result = rnaseq.log10_transformation(data, addition=0.0)
    assert np.isnan(result.loc['g1', 'C1'])
    assert np.isclose(result.loc['g2', 'C1'], 0.0)


# ---------------------------------------------------------------------------------
# scale_expression_by_sum
# ---------------------------------------------------------------------------------

def test_scale_expression_by_sum_columns(toy_rnaseq):
    result = rnaseq.scale_expression_by_sum(toy_rnaseq, axis=0, sum_value=1e6)
    assert np.allclose(result.sum(axis=0).values, 1e6)
    assert list(result.columns) == list(toy_rnaseq.columns)


def test_scale_expression_by_sum_rows(toy_rnaseq):
    result = rnaseq.scale_expression_by_sum(toy_rnaseq, axis=1, sum_value=1.0)
    assert np.allclose(result.sum(axis=1).values, 1.0)


def test_scale_expression_by_sum_preserves_proportions(toy_rnaseq):
    result = rnaseq.scale_expression_by_sum(toy_rnaseq, axis=0, sum_value=1.0)
    ratio_before = toy_rnaseq.iloc[0, 0] / toy_rnaseq.iloc[1, 0]
    ratio_after = result.iloc[0, 0] / result.iloc[1, 0]
    assert np.isclose(ratio_before, ratio_after)


# ---------------------------------------------------------------------------------
# divide_expression_by_max / by_mean
# ---------------------------------------------------------------------------------

def test_divide_expression_by_max_rowwise(toy_rnaseq):
    result = rnaseq.divide_expression_by_max(toy_rnaseq, axis=1)
    assert np.allclose(result.max(axis=1).values, 1.0)
    assert (result.values <= 1.0).all()


def test_divide_expression_by_max_columnwise(toy_rnaseq):
    result = rnaseq.divide_expression_by_max(toy_rnaseq, axis=0)
    assert np.allclose(result.max(axis=0).values, 1.0)


def test_divide_expression_by_mean_rowwise(toy_rnaseq):
    result = rnaseq.divide_expression_by_mean(toy_rnaseq, axis=1)
    expected = toy_rnaseq.values / toy_rnaseq.mean(axis=1).values[:, None]
    assert np.allclose(result.values, expected)


def test_divide_expression_handles_zero_denominator():
    data = pd.DataFrame({'C1': [0.0, 2.0], 'C2': [0.0, 4.0]}, index=['g1', 'g2'])
    by_max = rnaseq.divide_expression_by_max(data, axis=1)
    by_mean = rnaseq.divide_expression_by_mean(data, axis=1)
    # An all-zero gene must not produce NaN or inf
    assert np.isfinite(by_max.values).all()
    assert np.isfinite(by_mean.values).all()
    assert (by_max.loc['g1'] == 0.0).all()


# ---------------------------------------------------------------------------------
# add_complexes_to_expression
# ---------------------------------------------------------------------------------

def test_add_complexes_to_expression_min(toy_rnaseq):
    complexes = {'Protein-A&Protein-B': ['Protein-A', 'Protein-B']}
    result = rnaseq.add_complexes_to_expression(toy_rnaseq, complexes, agg_method='min')
    assert 'Protein-A&Protein-B' in result.index
    expected = toy_rnaseq.loc[['Protein-A', 'Protein-B']].min().values
    assert np.allclose(result.loc['Protein-A&Protein-B'].values, expected)


def test_add_complexes_to_expression_mean(toy_rnaseq):
    complexes = {'complex': ['Protein-A', 'Protein-B']}
    result = rnaseq.add_complexes_to_expression(toy_rnaseq, complexes, agg_method='mean')
    expected = toy_rnaseq.loc[['Protein-A', 'Protein-B']].mean().values
    assert np.allclose(result.loc['complex'].values, expected)


def test_add_complexes_to_expression_gmean(toy_rnaseq):
    complexes = {'complex': ['Protein-A', 'Protein-B']}
    result = rnaseq.add_complexes_to_expression(toy_rnaseq, complexes, agg_method='gmean')
    subset = toy_rnaseq.loc[['Protein-A', 'Protein-B']].values.astype(float)
    expected = np.exp(np.mean(np.log(subset), axis=0))
    assert np.allclose(result.loc['complex'].values, expected)


def test_add_complexes_to_expression_missing_subunit_gives_zeros(toy_rnaseq):
    complexes = {'complex': ['Protein-A', 'Not-A-Gene']}
    result = rnaseq.add_complexes_to_expression(toy_rnaseq, complexes)
    assert (result.loc['complex'] == 0).all()


def test_add_complexes_to_expression_accepts_sets(toy_rnaseq):
    complexes = {'complex': {'Protein-A', 'Protein-B'}}
    result = rnaseq.add_complexes_to_expression(toy_rnaseq, complexes, agg_method='min')
    assert 'complex' in result.index


def test_add_complexes_to_expression_rejects_bad_values(toy_rnaseq):
    with pytest.raises(ValueError):
        rnaseq.add_complexes_to_expression(toy_rnaseq, {'complex': 'Protein-A'})


def test_add_complexes_to_expression_does_not_modify_input(toy_rnaseq):
    before = toy_rnaseq.copy()
    rnaseq.add_complexes_to_expression(toy_rnaseq, {'c': ['Protein-A', 'Protein-B']})
    pd.testing.assert_frame_equal(toy_rnaseq, before)


# ---------------------------------------------------------------------------------
# _trimean
# ---------------------------------------------------------------------------------

def test_trimean_axis_semantics():
    x = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
    by_column = _trimean(x, axis=0)
    by_row = _trimean(x, axis=1)
    assert by_column.shape == (3,)
    assert by_row.shape == (4,)
    assert np.allclose(by_column, [5.5, 6.5, 7.5])
    assert np.allclose(by_row, [2., 5., 8., 11.])


def test_trimean_formula():
    values = np.array([[1.], [2.], [3.], [10.], [20.]])
    q1, q2, q3 = np.nanpercentile(values, [25, 50, 75], axis=0)
    assert np.allclose(_trimean(values, axis=0), 0.5 * q2 + 0.25 * (q1 + q3))


def test_trimean_ignores_nan():
    values = np.array([[1.], [np.nan], [3.]])
    result = _trimean(values, axis=0)
    assert np.isfinite(result).all()
    assert np.isclose(result[0], 2.0)


def test_trimean_of_a_single_value():
    assert np.isclose(_trimean(np.array([[7.]]), axis=0)[0], 7.0)


def test_trimean_is_more_robust_than_the_mean():
    clean = np.array([[1.], [2.], [3.], [4.], [5.]])
    with_outlier = np.array([[1.], [2.], [3.], [4.], [1000.]])
    mean_shift = abs(with_outlier.mean() - clean.mean())
    trimean_shift = abs(_trimean(with_outlier, axis=0)[0] - _trimean(clean, axis=0)[0])
    assert trimean_shift < mean_shift


# ---------------------------------------------------------------------------------
# aggregate_single_cells
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['average', 'nn_cell_fraction', 'trimean'])
def test_aggregate_single_cells_shape_and_labels(toy_single_cells, method):
    data, metadata = toy_single_cells
    result = rnaseq.aggregate_single_cells(data.T, metadata, barcode_col='barcodes',
                                           celltype_col='cell_types', method=method)
    assert list(result.index) == list(data.index)
    assert list(result.columns) == ['CT-1', 'CT-2', 'CT-3']


def test_aggregate_single_cells_average_values(toy_single_cells):
    data, metadata = toy_single_cells
    result = rnaseq.aggregate_single_cells(data.T, metadata, barcode_col='barcodes',
                                           celltype_col='cell_types', method='average')
    for cell_type in ['CT-1', 'CT-2', 'CT-3']:
        barcodes = metadata.loc[metadata['cell_types'] == cell_type, 'barcodes']
        expected = data[list(barcodes)].mean(axis=1)
        assert np.allclose(result[cell_type].values, expected.values)


def test_aggregate_single_cells_nn_cell_fraction_values(toy_single_cells):
    data, metadata = toy_single_cells
    result = rnaseq.aggregate_single_cells(data.T, metadata, barcode_col='barcodes',
                                           celltype_col='cell_types',
                                           method='nn_cell_fraction')
    assert result.values.min() >= 0.0 and result.values.max() <= 1.0
    for cell_type in ['CT-1', 'CT-2']:
        barcodes = list(metadata.loc[metadata['cell_types'] == cell_type, 'barcodes'])
        expected = (data[barcodes] > 0).sum(axis=1) / len(barcodes)
        assert np.allclose(result[cell_type].values, expected.values)


def test_aggregate_single_cells_trimean_values(toy_single_cells):
    data, metadata = toy_single_cells
    result = rnaseq.aggregate_single_cells(data.T, metadata, barcode_col='barcodes',
                                           celltype_col='cell_types', method='trimean')
    for cell_type in ['CT-1', 'CT-2', 'CT-3']:
        barcodes = list(metadata.loc[metadata['cell_types'] == cell_type, 'barcodes'])
        for gene in data.index:
            q1, q2, q3 = np.nanpercentile(data.loc[gene, barcodes].values, [25, 50, 75])
            assert np.isclose(result.loc[gene, cell_type], 0.5 * q2 + 0.25 * (q1 + q3))


def test_aggregate_single_cells_transposed_false_matches(toy_single_cells):
    data, metadata = toy_single_cells
    transposed = rnaseq.aggregate_single_cells(data.T, metadata, barcode_col='barcodes',
                                               celltype_col='cell_types', method='average')
    straight = rnaseq.aggregate_single_cells(data, metadata, barcode_col='barcodes',
                                             celltype_col='cell_types', method='average',
                                             transposed=False)
    pd.testing.assert_frame_equal(transposed, straight)


def test_aggregate_single_cells_gene_labels_survive_column_permutation(toy_single_cells):
    '''Guards against positional (rather than labelled) assignment of the results.'''
    data, metadata = toy_single_cells
    reference = rnaseq.aggregate_single_cells(data.T, metadata, barcode_col='barcodes',
                                              celltype_col='cell_types', method='trimean')
    permuted_genes = list(data.index)[::-1]
    permuted = rnaseq.aggregate_single_cells(data.loc[permuted_genes].T, metadata,
                                             barcode_col='barcodes',
                                             celltype_col='cell_types', method='trimean')
    for gene in data.index:
        for cell_type in reference.columns:
            assert np.isclose(reference.loc[gene, cell_type], permuted.loc[gene, cell_type])


def test_aggregate_single_cells_rejects_invalid_method(toy_single_cells):
    data, metadata = toy_single_cells
    with pytest.raises(AssertionError):
        rnaseq.aggregate_single_cells(data.T, metadata, method='not-a-method')


def test_aggregate_single_cells_requires_metadata(toy_single_cells):
    data, _ = toy_single_cells
    with pytest.raises(AssertionError):
        rnaseq.aggregate_single_cells(data.T, None)


def test_aggregate_single_cells_single_cell_per_type():
    data = pd.DataFrame({'b1': [5.0, 7.0], 'b2': [1.0, 2.0]}, index=['g1', 'g2'])
    metadata = pd.DataFrame({'barcodes': ['b1', 'b2'], 'cell_types': ['x', 'y']})
    result = rnaseq.aggregate_single_cells(data.T, metadata, method='trimean')
    assert np.allclose(result['x'].values, [5.0, 7.0])
    assert np.allclose(result['y'].values, [1.0, 2.0])
