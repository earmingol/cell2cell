# -*- coding: utf-8 -*-

'''Tests for cell2cell.preprocessing.cutoffs'''

import numpy as np
import pytest

from cell2cell.preprocessing import cutoffs


def test_get_local_percentile_cutoffs_values(toy_rnaseq):
    result = cutoffs.get_local_percentile_cutoffs(toy_rnaseq, percentile=0.75)
    assert list(result.columns) == ['value']
    assert list(result.index) == list(toy_rnaseq.index)
    expected = toy_rnaseq.quantile(0.75, axis=1).values
    assert np.allclose(result['value'].values, expected)


def test_get_local_percentile_cutoffs_is_per_gene(toy_rnaseq):
    '''Each gene gets its own cutoff, so the values must not be all identical.'''
    result = cutoffs.get_local_percentile_cutoffs(toy_rnaseq, percentile=0.5)
    assert result['value'].nunique() > 1


def test_get_global_percentile_cutoffs_is_a_single_value(toy_rnaseq):
    result = cutoffs.get_global_percentile_cutoffs(toy_rnaseq, percentile=0.75)
    assert list(result.columns) == ['value']
    assert result['value'].nunique() == 1
    expected = np.percentile(toy_rnaseq.values, 75)
    assert np.allclose(result['value'].values, expected)


def test_get_constant_cutoff(toy_rnaseq):
    result = cutoffs.get_constant_cutoff(toy_rnaseq, constant_cutoff=7)
    assert (result['value'] == 7).all()
    assert list(result.index) == list(toy_rnaseq.index)


@pytest.mark.parametrize('cutoff_type,parameter', [('local_percentile', 0.75),
                                                   ('global_percentile', 0.75),
                                                   ('constant_value', 12)])
def test_get_cutoffs_dispatches(toy_rnaseq, cutoff_type, parameter):
    parameters = {'type': cutoff_type, 'parameter': parameter}
    result = cutoffs.get_cutoffs(toy_rnaseq, parameters, verbose=False)
    assert list(result.columns) == ['value']
    assert result.shape[0] == toy_rnaseq.shape[0]


def test_get_cutoffs_matches_the_direct_functions(toy_rnaseq):
    direct = cutoffs.get_local_percentile_cutoffs(toy_rnaseq, percentile=0.6)
    through = cutoffs.get_cutoffs(toy_rnaseq,
                                  {'type': 'local_percentile', 'parameter': 0.6},
                                  verbose=False)
    assert np.allclose(direct['value'].values, through['value'].values)


def test_get_cutoffs_rejects_unknown_type(toy_rnaseq):
    with pytest.raises(ValueError):
        cutoffs.get_cutoffs(toy_rnaseq, {'type': 'nonsense', 'parameter': 1},
                            verbose=False)


def test_cutoffs_do_not_modify_input(toy_rnaseq):
    before = toy_rnaseq.copy()
    cutoffs.get_local_percentile_cutoffs(toy_rnaseq)
    cutoffs.get_global_percentile_cutoffs(toy_rnaseq)
    cutoffs.get_constant_cutoff(toy_rnaseq)
    assert toy_rnaseq.equals(before)
