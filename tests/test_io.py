# -*- coding: utf-8 -*-

'''Tests for cell2cell.io'''

import os

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.io import directories, read_data, save_data


# ---------------------------------------------------------------------------------
# directories
# ---------------------------------------------------------------------------------

def test_create_directory(tmp_path):
    target = tmp_path / 'new-folder'
    directories.create_directory(str(target))
    assert target.is_dir()


def test_create_directory_is_idempotent(tmp_path):
    target = tmp_path / 'folder'
    directories.create_directory(str(target))
    directories.create_directory(str(target))
    assert target.is_dir()


def test_get_files_from_directory_is_naturally_sorted(tmp_path):
    for name in ['f10.csv', 'f2.csv', 'f1.csv']:
        (tmp_path / name).write_text('')
    files = directories.get_files_from_directory(str(tmp_path))
    assert files == ['f1.csv', 'f2.csv', 'f10.csv']


def test_get_files_from_directory_with_full_paths(tmp_path):
    (tmp_path / 'a.csv').write_text('')
    files = directories.get_files_from_directory(str(tmp_path), dir_in_filepath=True)
    assert files == [str(tmp_path) + '/a.csv']
    assert os.path.isfile(files[0])


def test_get_files_from_directory_is_reproducible(tmp_path):
    for name in ['z.csv', 'a.csv', 'm.csv']:
        (tmp_path / name).write_text('')
    assert (directories.get_files_from_directory(str(tmp_path)) ==
            directories.get_files_from_directory(str(tmp_path)))


# ---------------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------------

def test_pickle_roundtrip_dataframe(tmp_path, toy_rnaseq):
    filename = str(tmp_path / 'data.pkl')
    save_data.export_variable_with_pickle(toy_rnaseq, filename)
    loaded = read_data.load_variable_with_pickle(filename)
    pd.testing.assert_frame_equal(loaded, toy_rnaseq)


def test_pickle_roundtrip_dictionary(tmp_path):
    variable = {'a': [1, 2, 3], 'b': 'text'}
    filename = str(tmp_path / 'var.pkl')
    save_data.export_variable_with_pickle(variable, filename)
    assert read_data.load_variable_with_pickle(filename) == variable


# ---------------------------------------------------------------------------------
# load_table and friends
# ---------------------------------------------------------------------------------

def test_load_table_auto_infers_the_separator_from_the_extension(tmp_path, toy_rnaseq):
    '''format="auto" sets sep=","  for .csv and sep="\\t" for .tsv/.txt.'''
    filename = tmp_path / 'table.csv'
    toy_rnaseq.to_csv(filename)
    loaded = read_data.load_table(str(filename), format='auto', index_col=0, verbose=False)
    assert list(loaded.columns) == list(toy_rnaseq.columns)
    assert np.allclose(loaded.values, toy_rnaseq.values)


def test_load_table_explicit_format_keeps_the_given_separator(tmp_path, toy_rnaseq):
    '''With an explicit format, `sep` is NOT inferred and defaults to a tab.'''
    filename = tmp_path / 'table.csv'
    toy_rnaseq.to_csv(filename)
    loaded = read_data.load_table(str(filename), format='csv', sep=',', index_col=0,
                                 verbose=False)
    assert list(loaded.columns) == list(toy_rnaseq.columns)


def test_load_table_returns_none_without_a_filename():
    assert read_data.load_table(None) is None


def test_load_table_tsv(tmp_path, toy_rnaseq):
    filename = tmp_path / 'table.tsv'
    toy_rnaseq.to_csv(filename, sep='\t')
    loaded = read_data.load_table(str(filename), format='auto', sep='\t', index_col=0,
                                 verbose=False)
    assert loaded.shape == toy_rnaseq.shape


def test_load_table_excel(tmp_path, toy_rnaseq):
    filename = tmp_path / 'table.xlsx'
    toy_rnaseq.to_excel(filename)
    loaded = read_data.load_table(str(filename), format='excel', index_col=0,
                                 verbose=False)
    assert loaded.shape == toy_rnaseq.shape


def test_load_table_returns_none_for_an_unknown_format(tmp_path):
    '''An unrecognized format returns None rather than raising.'''
    filename = tmp_path / 'table.weird'
    filename.write_text('a,b\n1,2\n')
    assert read_data.load_table(str(filename), format='nonsense', verbose=False) is None


def test_load_tables_from_directory(tmp_path, toy_rnaseq):
    for name in ['s1', 's2']:
        toy_rnaseq.to_csv(tmp_path / '{}.csv'.format(name))
    tables = read_data.load_tables_from_directory(str(tmp_path), extension='csv',
                                                 sep=',', index_col=0, verbose=False)
    assert set(tables.keys()) == {'s1', 's2'}
    for frame in tables.values():
        assert frame.shape == toy_rnaseq.shape


def test_load_rnaseq(tmp_path, toy_rnaseq):
    frame = toy_rnaseq.reset_index()
    filename = tmp_path / 'rnaseq.csv'
    frame.to_csv(filename, index=False)
    loaded = read_data.load_rnaseq(str(filename), gene_column='gene_id', format='auto',
                                  verbose=False)
    assert list(loaded.index) == list(toy_rnaseq.index)


def test_load_metadata(tmp_path, toy_metadata):
    filename = tmp_path / 'metadata.csv'
    toy_metadata.to_csv(filename, index=False)
    loaded = read_data.load_metadata(str(filename), format='auto')
    assert 'Groups' in loaded.columns


def test_load_ppi(tmp_path, toy_ppi):
    filename = tmp_path / 'ppi.csv'
    toy_ppi.to_csv(filename, index=False)
    loaded = read_data.load_ppi(str(filename), interaction_columns=('A', 'B'),
                                format='auto', verbose=False)
    assert list(loaded.columns) == ['A', 'B', 'score']


def test_load_cutoffs(tmp_path, toy_rnaseq):
    cutoffs = c2c.preprocessing.get_constant_cutoff(toy_rnaseq, constant_cutoff=5)
    frame = cutoffs.reset_index()
    filename = tmp_path / 'cutoffs.csv'
    frame.to_csv(filename, index=False)
    loaded = read_data.load_cutoffs(str(filename), gene_column='gene_id', format='auto',
                                    verbose=False)
    assert 'value' in loaded.columns


# ---------------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------------

def test_load_tensor_roundtrip(tmp_path, factorized_tensor):
    filename = str(tmp_path / 'tensor.pkl')
    save_data.export_variable_with_pickle(factorized_tensor, filename)
    loaded = read_data.load_tensor(filename)
    assert np.allclose(np.asarray(loaded.tensor), np.asarray(factorized_tensor.tensor))
    assert [list(o) for o in loaded.order_names] == \
           [list(o) for o in factorized_tensor.order_names]


def test_load_tensor_factors_roundtrip(tmp_path, factorized_tensor):
    filename = str(tmp_path / 'factors.xlsx')
    factorized_tensor.export_factor_loadings(filename)
    loaded = read_data.load_tensor_factors(filename)
    assert list(loaded.keys()) == list(factorized_tensor.factors.keys())
    for key, frame in loaded.items():
        expected = factorized_tensor.factors[key]
        assert list(frame.index) == list(expected.index)
        assert np.allclose(frame.values, expected.values)


# ---------------------------------------------------------------------------------
# load_table: the rest of the format auto-detection
# ---------------------------------------------------------------------------------

def test_load_table_auto_detects_a_txt_file(tmp_path, toy_rnaseq):
    '''.txt is read as tab-separated, like .tsv.'''
    filename = tmp_path / 'table.txt'
    toy_rnaseq.to_csv(filename, sep='\t')
    loaded = read_data.load_table(str(filename), format='auto', index_col=0, verbose=False)
    assert loaded.shape == toy_rnaseq.shape


def test_load_table_auto_detects_an_excel_file(tmp_path, toy_rnaseq):
    filename = tmp_path / 'table.xlsx'
    toy_rnaseq.to_excel(filename)
    loaded = read_data.load_table(str(filename), format='auto', index_col=0, verbose=False)
    assert loaded.shape == toy_rnaseq.shape


def test_load_table_auto_detects_a_gzipped_file(tmp_path, toy_rnaseq):
    '''.gz implies a gzip-compressed, tab-separated table.'''
    filename = tmp_path / 'table.tsv.gz'
    toy_rnaseq.to_csv(filename, sep='\t', compression='gzip')
    loaded = read_data.load_table(str(filename), format='auto', index_col=0, verbose=False)
    assert loaded.shape == toy_rnaseq.shape


def test_load_table_reports_what_it_loaded(tmp_path, toy_rnaseq, capsys):
    filename = tmp_path / 'table.csv'
    toy_rnaseq.to_csv(filename)
    read_data.load_table(str(filename), format='auto', index_col=0, verbose=True)
    assert 'was correctly loaded' in capsys.readouterr().out


def test_load_tables_from_directory_with_compressed_files(tmp_path, toy_rnaseq):
    for name in ['s1', 's2']:
        toy_rnaseq.to_csv(tmp_path / '{}.csv.gzip'.format(name), compression='gzip')
    tables = read_data.load_tables_from_directory(str(tmp_path), extension='csv',
                                                  sep=',', compression='gzip',
                                                  index_col=0, verbose=False)
    assert set(tables.keys()) == {'s1', 's2'}


def test_load_tables_from_directory_rejects_an_unknown_compression(tmp_path):
    with pytest.raises(AssertionError):
        read_data.load_tables_from_directory(str(tmp_path), extension='csv',
                                             compression='nonsense')


# ---------------------------------------------------------------------------------
# The loaders that post-process what load_table returns
# ---------------------------------------------------------------------------------

def test_load_rnaseq_can_log_transform(tmp_path, toy_rnaseq):
    frame = toy_rnaseq.reset_index()
    filename = tmp_path / 'rnaseq.csv'
    frame.to_csv(filename, index=False)
    loaded = read_data.load_rnaseq(str(filename), gene_column='gene_id', format='auto',
                                   log_transformation=True, verbose=False)
    assert np.allclose(loaded.values, np.log10(toy_rnaseq.values + 1e-6))


def test_load_metadata_can_index_and_filter_by_cell(tmp_path, toy_metadata):
    filename = tmp_path / 'metadata.csv'
    toy_metadata.to_csv(filename, index=False)
    labels = list(toy_metadata['#SampleID'])[:2]
    loaded = read_data.load_metadata(str(filename), cell_labels=labels,
                                     index_col='#SampleID', format='auto')
    assert list(loaded.index) == labels


def test_load_cutoffs_without_a_gene_column(tmp_path, toy_rnaseq):
    '''Without `gene_column` the first column becomes the index.'''
    cutoffs = c2c.preprocessing.get_constant_cutoff(toy_rnaseq, constant_cutoff=5)
    filename = tmp_path / 'cutoffs.csv'
    cutoffs.reset_index().to_csv(filename, index=False)
    loaded = read_data.load_cutoffs(str(filename), gene_column=None, format='auto',
                                    verbose=False)
    assert list(loaded.index) == list(toy_rnaseq.index)
    assert list(loaded.columns) == ['value']


def test_load_ppi_filters_by_score_and_by_the_genes_measured(tmp_path, toy_ppi, toy_rnaseq):
    filename = tmp_path / 'ppi.csv'
    toy_ppi.to_csv(filename, index=False)
    genes = list(toy_rnaseq.index)[:3]
    loaded = read_data.load_ppi(str(filename), interaction_columns=('A', 'B'),
                                score='score', rnaseq_genes=genes, format='auto',
                                verbose=False)
    assert set(loaded['A']).issubset(set(genes))
    assert set(loaded['B']).issubset(set(genes))


def test_load_ppi_with_complexes(tmp_path, toy_ppi_complex):
    filename = tmp_path / 'ppi.csv'
    toy_ppi_complex.to_csv(filename, index=False)
    loaded = read_data.load_ppi(str(filename), interaction_columns=('A', 'B'),
                                complex_sep='&', format='auto', verbose=False)
    assert list(loaded.columns) == ['A', 'B', 'score']


# ---------------------------------------------------------------------------------
# Gene ontology files
# ---------------------------------------------------------------------------------

def test_load_go_terms(tiny_obo):
    graph = read_data.load_go_terms(tiny_obo, verbose=False)
    assert 'GO:0000001' in graph.nodes()


def test_load_go_annotations_renames_the_columns(tiny_gaf):
    '''Only four of the GAF columns are kept, under shorter names.'''
    annotations = read_data.load_go_annotations(tiny_gaf, experimental_evidence=False,
                                                verbose=False)
    assert list(annotations.columns) == ['db', 'Gene', 'Name', 'GO']
    assert list(annotations['Gene']) == ['Protein-A-id', 'Protein-B-id', 'Protein-C-id']
    assert annotations['GO'].str.startswith('GO:').all()


def test_load_go_annotations_can_keep_only_experimental_evidence(tiny_gaf):
    annotations = read_data.load_go_annotations(tiny_gaf, experimental_evidence=True,
                                                verbose=False)
    assert list(annotations['Name']) == ['Protein-A', 'Protein-B']


# ---------------------------------------------------------------------------------
# load_tensor on a tensor that carries a mask
# ---------------------------------------------------------------------------------

def test_load_tensor_restores_the_mask_and_the_nan_positions(tmp_path):
    '''A tensor with missing values keeps `mask`, `loc_nans` and `loc_zeros`, which
    are converted back into tensorly tensors on load.'''
    values = np.arange(2 * 2 * 3 * 3, dtype=float).reshape((2, 2, 3, 3))
    values[0, 0, 0, 0] = np.nan
    names = [['Context-1', 'Context-2'], ['LR-1', 'LR-2'],
             ['C1', 'C2', 'C3'], ['C1', 'C2', 'C3']]
    tensor = c2c.tensor.PreBuiltTensor(tensor=values, order_names=names,
                                       mask=(~np.isnan(values)).astype(int))
    filename = str(tmp_path / 'masked-tensor.pkl')
    save_data.export_variable_with_pickle(tensor, filename)

    loaded = read_data.load_tensor(filename)
    assert loaded.mask is not None
    assert np.allclose(np.asarray(loaded.mask), np.asarray(tensor.mask))
    assert np.allclose(np.asarray(loaded.loc_nans), np.asarray(tensor.loc_nans))
    assert np.allclose(np.asarray(loaded.loc_zeros), np.asarray(tensor.loc_zeros))
