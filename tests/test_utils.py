# -*- coding: utf-8 -*-

'''Tests for cell2cell.utils'''

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cell2cell.utils import networks, parallel_computing


# ---------------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------------

@pytest.fixture
def adjacency(toy_distance):
    '''A weighted adjacency matrix derived from the toy distances.'''
    similarity = 1 - toy_distance / toy_distance.values.max()
    np.fill_diagonal(similarity.values, 0.0)
    return similarity


def test_generate_network_from_adjacency_networkx(adjacency):
    graph = networks.generate_network_from_adjacency(adjacency, package='networkx')
    assert isinstance(graph, nx.Graph)
    assert set(graph.nodes()) == set(adjacency.index)


def test_generated_network_keeps_the_edge_weights(adjacency):
    graph = networks.generate_network_from_adjacency(adjacency, package='networkx')
    for node_a, node_b, data in graph.edges(data=True):
        assert np.isclose(data['weight'], adjacency.loc[node_a, node_b])


def test_generate_network_from_adjacency_rejects_other_packages(adjacency):
    with pytest.raises((ValueError, NotImplementedError)):
        networks.generate_network_from_adjacency(adjacency, package='igraph')


def test_export_network_to_gephi_excel(adjacency, tmp_path):
    filename = tmp_path / 'network.xlsx'
    networks.export_network_to_gephi(adjacency, str(filename), format='excel')
    assert filename.exists() and filename.stat().st_size > 0


def test_export_network_to_gephi_csv(adjacency, tmp_path):
    filename = tmp_path / 'network.csv'
    networks.export_network_to_gephi(adjacency, str(filename), format='csv')
    assert filename.exists()
    written = pd.read_csv(filename)
    assert written.shape[0] > 0


def test_export_network_to_gephi_accepts_a_graph(adjacency, tmp_path):
    graph = networks.generate_network_from_adjacency(adjacency, package='networkx')
    filename = tmp_path / 'from-graph.csv'
    networks.export_network_to_gephi(graph, str(filename), format='csv')
    assert filename.exists()


def test_export_network_to_cytoscape(adjacency, tmp_path):
    filename = tmp_path / 'network.cyjs'
    networks.export_network_to_cytoscape(adjacency, str(filename))
    assert filename.exists()
    import json
    with open(filename) as handle:
        content = json.load(handle)
    assert 'elements' in content


# ---------------------------------------------------------------------------------
# parallel_computing
# ---------------------------------------------------------------------------------

def test_agents_number_positive():
    assert parallel_computing.agents_number(1) == 1
    assert parallel_computing.agents_number(2) >= 1


def test_agents_number_uses_all_cores_for_minus_one():
    import multiprocessing
    assert parallel_computing.agents_number(-1) == multiprocessing.cpu_count()


def test_agents_number_never_exceeds_the_cpu_count():
    import multiprocessing
    total = multiprocessing.cpu_count()
    assert parallel_computing.agents_number(total * 10) <= total


def test_agents_number_handles_negative_values():
    import multiprocessing
    total = multiprocessing.cpu_count()
    result = parallel_computing.agents_number(-2)
    assert 1 <= result <= total
