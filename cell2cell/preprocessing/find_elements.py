# -*- coding: utf-8 -*-

from __future__ import absolute_import

from collections import defaultdict

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