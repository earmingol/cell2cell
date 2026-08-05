# -*- coding: utf-8 -*-

'''Tests for cell2cell.preprocessing.find_elements'''

import numpy as np

from cell2cell.preprocessing import find_elements


def test_find_duplicates():
    result = find_elements.find_duplicates(['a', 'b', 'a', 'c', 'b', 'a'])
    assert result == {'a': [0, 2, 5], 'b': [1, 4]}


def test_find_duplicates_without_duplicates():
    assert find_elements.find_duplicates(['a', 'b', 'c']) == {}


def test_find_duplicates_on_empty_list():
    assert find_elements.find_duplicates([]) == {}


def test_get_element_abundances_fractions():
    result = find_elements.get_element_abundances([['a', 'b'], ['b', 'c'], ['b']])
    assert np.isclose(result['b'], 1.0)
    assert np.isclose(result['a'], 1 / 3)
    assert np.isclose(result['c'], 1 / 3)


def test_get_element_abundances_ignores_repeats_within_a_list():
    result = find_elements.get_element_abundances([['a', 'a', 'a'], ['b']])
    assert np.isclose(result['a'], 0.5)


def test_get_element_abundances_key_order_is_first_appearance():
    result = find_elements.get_element_abundances([['z', 'a'], ['m', 'z']])
    assert list(result.keys()) == ['z', 'a', 'm']


def test_get_element_abundances_single_list():
    result = find_elements.get_element_abundances([['a', 'b']])
    assert np.isclose(result['a'], 1.0)
    assert np.isclose(result['b'], 1.0)


def test_get_elements_over_fraction():
    abundances = {'a': 1.0, 'b': 0.5, 'c': 0.2}
    assert find_elements.get_elements_over_fraction(abundances, 0.5) == ['a', 'b']
    assert find_elements.get_elements_over_fraction(abundances, 1.0) == ['a']
    assert find_elements.get_elements_over_fraction(abundances, 0.0) == ['a', 'b', 'c']


def test_get_elements_over_fraction_keeps_the_dict_order():
    abundances = {'z': 1.0, 'a': 1.0}
    assert find_elements.get_elements_over_fraction(abundances, 0.5) == ['z', 'a']


def test_get_elements_over_fraction_can_return_nothing():
    assert find_elements.get_elements_over_fraction({'a': 0.1}, 0.9) == []


def test_element_abundance_pipeline_is_reproducible():
    lists = [['b', 'a', 'c'], ['c', 'b', 'z'], ['b', 'c', 'q']]
    first = find_elements.get_elements_over_fraction(
        find_elements.get_element_abundances(lists), 0.5)
    second = find_elements.get_elements_over_fraction(
        find_elements.get_element_abundances(lists), 0.5)
    assert first == second
