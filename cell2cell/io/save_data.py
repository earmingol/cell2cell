# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pickle
import networkx as nx


def export_variable(variable, filename):
    '''
    Export a large size variable in a python readable way using pickle.
    '''

    max_bytes = 2 ** 31 - 1

    bytes_out = pickle.dumps(variable)
    with open(filename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def network2gephi(cci_network, filename, format='excel'):
    '''
    Export a CCI network (networkx object) into a spreadsheet readable by Gephi.
    '''
    gephi_df = nx.to_pandas_edgelist(cci_network)
    gephi_df = gephi_df.assign(Type='Undirected')
    gephi_df = gephi_df[['source', 'target', 'Type', 'weight']]
    gephi_df.columns = ['Source', 'Target', 'Type', 'Weight']
    if format == 'excel':
        gephi_df.to_excel(filename, sheet_name='Edges')
    elif format == 'csv':
        gephi_df.to_csv(filename, sep=',')
    elif format == 'tsv':
        gephi_df.to_csv(filename, sep='\t')
    else:
        raise ValueError("Format not supported.")