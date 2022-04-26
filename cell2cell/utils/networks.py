# -*- coding: utf-8 -*-

from __future__ import absolute_import

import networkx as nx


def generate_network_from_adjacency(adjacency_matrix, package='networkx'):
    '''
    Generates a network or graph object from an adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : pandas.DataFrame
        An adjacency matrix, where in rows and columns are nodes
        and values represents a weight for the respective edge.

    package : str, default='networkx'
        Package or python library to built the network.
        Implemented optios are {'networkx'}. Soon will be
        available for 'igraph'.

    Returns
    -------
    network : graph-like
        A graph object built with a python-library for networks.
    '''
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


def export_network_to_gephi(network, filename, format='excel', network_type='Undirected'):
    '''
    Exports a network into a spreadsheet that is readable
    by the software Gephi.

    Parameters
    ----------
    network : networkx.Graph, networkx.DiGraph or a pandas.DataFrame
        A networkx Graph or Directed Graph, or an adjacency matrix,
        where in rows and columns are nodes and values represents a
        weight for the respective edge.

    filename : str, default=None
        Path to save the network into a Gephi-readable format.

    format : str, default='excel'
        Format to export the spreadsheet. Options are:

        - 'excel' : An excel file, either .xls or .xlsx
        - 'csv' : Comma separated value format
        - 'tsv' : Tab separated value format

    network_type : str, default='Undirected'
        Type of edges in the network. They could be either
        'Undirected' or 'Directed'.
    '''
    # This allows to pass a network directly or an adjacency matrix
    if type(network) != nx.classes.graph.Graph:
        network = generate_network_from_adjacency(network,
                                                  package='networkx')

    gephi_df = nx.to_pandas_edgelist(network)
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


def export_network_to_cytoscape(network, filename):
    '''
    Exports a network into a spreadsheet that is readable
    by the software Gephi.

    Parameters
    ----------
    network : networkx.Graph, networkx.DiGraph or a pandas.DataFrame
        A networkx Graph or Directed Graph, or an adjacency matrix,
        where in rows and columns are nodes and values represents a
        weight for the respective edge.

    filename : str, default=None
        Path to save the network into a Cytoscape-readable format
        (JSON file in this case). E.g. '/home/user/network.json'
    '''
    # This allows to pass a network directly or an adjacency matrix
    if type(network) != nx.classes.graph.Graph:
        network = generate_network_from_adjacency(network,
                                                  package='networkx')

    data = nx.readwrite.json_graph.cytoscape.cytoscape_data(network)

    # Export
    import json
    json_str = json.dumps(data)
    with open(filename, 'w') as outfile:
        outfile.write(json_str)