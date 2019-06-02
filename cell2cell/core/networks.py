# -*- coding: utf-8 -*-

from __future__ import absolute_import

import networkx as nx


def network_from_adjacency(adjacency_matrix, package='networkx'):
    if package == 'networkx':
        network = nx.from_pandas_adjacency(adjacency_matrix)
    elif package == 'igraph':
        raise NotImplementedError("iGraph is not implemented yet.")
    else:
        raise NotImplementedError("Network using package {} not implemented".format(package))
    return network