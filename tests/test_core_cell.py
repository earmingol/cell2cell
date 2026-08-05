# -*- coding: utf-8 -*-

'''Tests for cell2cell.core.cell

NOTE: `Cell._id` is a class attribute that only ever increases (`__del__` decrements
`_id_counter`, not `_id`), so no test here asserts an absolute `cell.id` value -- that
would make the results depend on the execution order of the whole suite.
'''

import numpy as np
import pandas as pd
import pytest

from cell2cell.core import cell as cell_module


def test_cell_takes_its_type_from_the_column_name(toy_rnaseq):
    instance = cell_module.Cell(toy_rnaseq[['C3']], verbose=False)
    assert instance.type == 'C3'


def test_cell_renames_the_expression_column(toy_rnaseq):
    instance = cell_module.Cell(toy_rnaseq[['C1']], verbose=False)
    assert list(instance.rnaseq_data.columns) == ['value']
    assert np.allclose(instance.rnaseq_data['value'].values, toy_rnaseq['C1'].values)


def test_cell_copies_the_expression_data(toy_rnaseq):
    data = toy_rnaseq[['C1']].copy()
    instance = cell_module.Cell(data, verbose=False)
    data.iloc[0, 0] = 99999
    assert not np.isclose(instance.rnaseq_data['value'].iloc[0], 99999)


def test_cell_ids_are_unique(toy_rnaseq):
    first = cell_module.Cell(toy_rnaseq[['C1']], verbose=False)
    second = cell_module.Cell(toy_rnaseq[['C2']], verbose=False)
    assert first.id != second.id


def test_cell_str_contains_the_type(toy_rnaseq):
    instance = cell_module.Cell(toy_rnaseq[['C4']], verbose=False)
    assert 'C4' in str(instance)


def test_get_cells_from_rnaseq_builds_one_cell_per_column(toy_rnaseq):
    cells = cell_module.get_cells_from_rnaseq(toy_rnaseq, verbose=False)
    assert set(cells.keys()) == set(toy_rnaseq.columns)
    for name, instance in cells.items():
        assert instance.type == name


def test_get_cells_from_rnaseq_with_a_subset(toy_rnaseq):
    cells = cell_module.get_cells_from_rnaseq(toy_rnaseq, cell_columns=['C1', 'C3'],
                                              verbose=False)
    assert set(cells.keys()) == {'C1', 'C3'}


def test_get_cells_from_rnaseq_expression_matches_the_source(toy_rnaseq, toy_cells):
    for name, instance in toy_cells.items():
        assert np.allclose(instance.rnaseq_data['value'].values,
                           toy_rnaseq[name].values)


def test_get_cells_from_rnaseq_preserves_gene_order(toy_rnaseq, toy_cells):
    for instance in toy_cells.values():
        assert list(instance.rnaseq_data.index) == list(toy_rnaseq.index)
