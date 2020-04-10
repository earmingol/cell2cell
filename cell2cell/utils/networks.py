# -*- coding: utf-8 -*-

from __future__ import absolute_import

import networkx as nx

def generate_network_from_adjacency(adjacency_matrix, package='networkx'):
    if package == 'networkx':
        network = nx.from_pandas_adjacency(adjacency_matrix)
    elif package == 'igraph':
        # A = adjacency_matrix.values
        # network = igraph.Graph.Weighted_Adjacency((A > 0).tolist(), mode=igraph.ADJ_UNDIRECTED)
        #
        # # Add edge weights and node labels.
        # network.es['weight'] = A[A.nonzero()]
        # network.vs['label'] = list(adjacency_matrix.columns)
        #
        # Warning("iGraph functionalities are not completely implemented yet.")
        raise NotImplementedError("Network using package {} not implemented".format(package))
    else:
        raise NotImplementedError("Network using package {} not implemented".format(package))
    return network


def export_network_to_gephi(cci_matrix, filename, format='excel', network_type='Undirected'):
    '''
    Export a CCI matrix into a spreadsheet readable by Gephi.
    '''
    # This allows to pass a network directly or an adjacency matrix
    if type(cci_matrix) != nx.classes.graph.Graph:
        cci_network = generate_network_from_adjacency(cci_matrix,
                                                      package='networkx')
    else:
        cci_network = cci_matrix

    gephi_df = nx.to_pandas_edgelist(cci_network)
    gephi_df = gephi_df.assign(Type=network_type)
    # When weight is not in the network
    if ('weight' not in gephi_df.columns):
        gephi_df = gephi_df.assign(weight=1)

    # Transform column names
    gephi_df = gephi_df[['source', 'target', 'Type', 'weight']]
    gephi_df.columns = [c.capitalize() for c in gephi_df.columns]

    # Save with different formats
    if format == 'excel':
        gephi_df.to_excel(filename, sheet_name='Edges', index=False)
    elif format == 'csv':
        gephi_df.to_csv(filename, sep=',', index=False)
    elif format == 'tsv':
        gephi_df.to_csv(filename, sep='\t', index=False)
    else:
        raise ValueError("Format not supported.")