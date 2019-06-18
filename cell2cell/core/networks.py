# -*- coding: utf-8 -*-

from __future__ import absolute_import

import networkx as nx
import igraph


def network_from_adjacency(adjacency_matrix, package='networkx'):
    if package == 'networkx':
        network = nx.from_pandas_adjacency(adjacency_matrix)
    elif package == 'igraph':
        A = adjacency_matrix.values
        network = igraph.Graph.Weighted_Adjacency((A > 0).tolist(), mode=igraph.ADJ_UNDIRECTED)
        # Add edge weights and node labels.
        network.es['weight'] = A[A.nonzero()]
        network.vs['label'] = list(adjacency_matrix.columns)  # or a.index/a.columns

        Warning("iGraph functionalities are not completely implemented yet.")
    else:
        raise NotImplementedError("Network using package {} not implemented".format(package))
    return network