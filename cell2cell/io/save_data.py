# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pickle


def export_variable_with_pickle(variable, filename):
    '''Exports a large size variable in a python readable way
    using pickle.

    Parameters
    ----------
    variable : a python variable
        Variable to export

    filename : str
        Complete path to the file wherein the variable will be
        stored. For example:
        /home/user/variable.pkl
    '''

    max_bytes = 2 ** 31 - 1

    bytes_out = pickle.dumps(variable)
    with open(filename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])
    print(filename, ' was correctly saved.')