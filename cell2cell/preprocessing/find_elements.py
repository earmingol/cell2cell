# -*- coding: utf-8 -*-

from __future__ import absolute_import

import itertools
from collections import defaultdict, Counter

def find_duplicates(element_list):
    '''Function based on: https://stackoverflow.com/a/5419576/12032899
    Finds duplicate items and list their index location.

    Parameters
    ----------
    element_list : list
        List of elements

    Returns
    -------
    duplicate_dict : dict
        Dictionary with duplicate items. Keys are the items, and values
        are lists with the respective indexes where they are.
    '''
    tally = defaultdict(list)
    for i,item in enumerate(element_list):
        tally[item].append(i)

    duplicate_dict = {key : locs for key,locs in tally.items()
                            if len(locs)>1}
    return duplicate_dict


def get_element_abundances(element_lists):
    '''Computes the fraction of occurrence of each element
    in a list of lists.

    Parameters
    ----------
    element_lists : list
        List of lists of elements. Elements will be
        counted only once in each of the lists.

    Returns
    -------
    abundance_dict : dict
        Dictionary containing the number of times that an
        element was present, divided by the total number of
        lists in `element_lists`.
    '''
    abundance_dict = Counter(itertools.chain(*map(set, element_lists)))
    total = len(element_lists)
    abundance_dict = {k : v/total for k, v in abundance_dict.items()}
    return abundance_dict


def get_elements_over_fraction(abundance_dict, fraction):
    '''Obtains a list of elements with the
    fraction of occurrence at least the threshold.

    Parameters
    ----------
    abundance_dict : dict
        Dictionary containing the number of times that an
        element was present, divided by the total number of
        possible occurrences.

    fraction : float
        Threshold to filter the elements. Elements with at least
        this threshold will be included.

    Returns
    -------
    elements : list
        List of elements that met the fraction criteria.
    '''
    elements = [k for k, v in abundance_dict.items() if v >= fraction]
    return elements