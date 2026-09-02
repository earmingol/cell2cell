# -*- coding: utf-8 -*-

'''Tests for cell2cell.utils'''

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cell2cell.preprocessing import zero_diagonal
from cell2cell.utils import networks, parallel_computing


# ---------------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------------

@pytest.fixture
def adjacency(toy_distance):
    '''A weighted adjacency matrix derived from the toy distances.'''
    return zero_diagonal(1 - toy_distance / toy_distance.values.max())


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


def test_agents_number_of_zero_is_one():
    assert parallel_computing.agents_number(0) == 1


def test_agents_number_clamps_a_very_negative_request_to_one():
    '''n_jobs counts back from the number of cores, and never below 1.'''
    import multiprocessing
    assert parallel_computing.agents_number(-10 * multiprocessing.cpu_count()) == 1


def test_parallel_spatial_ccis_is_a_placeholder():
    '''Not implemented yet; it must at least not raise.'''
    assert parallel_computing.parallel_spatial_ccis(None) is None


@pytest.mark.parametrize('extension,format', [('xlsx', 'excel'), ('csv', 'csv'),
                                              ('tsv', 'tsv')])
def test_export_network_to_gephi_writes_every_supported_format(adjacency, tmp_path,
                                                               extension, format):
    graph = networks.generate_network_from_adjacency(adjacency, package='networkx')
    filename = tmp_path / 'network.{}'.format(extension)
    networks.export_network_to_gephi(graph, str(filename), format=format)
    assert filename.exists() and filename.stat().st_size > 0


def test_export_network_to_gephi_rejects_an_unknown_format(adjacency, tmp_path):
    graph = networks.generate_network_from_adjacency(adjacency, package='networkx')
    with pytest.raises(ValueError):
        networks.export_network_to_gephi(graph, str(tmp_path / 'n.txt'), format='nonsense')


def test_export_network_to_gephi_defaults_the_weight_when_there_is_none(tmp_path):
    '''An unweighted network still gets a Weight column, filled with 1.'''
    graph = nx.Graph()
    graph.add_edges_from([('a', 'b'), ('b', 'c')])
    filename = tmp_path / 'network.csv'
    networks.export_network_to_gephi(graph, str(filename), format='csv')
    edges = pd.read_csv(filename)
    assert list(edges.columns) == ['Source', 'Target', 'Type', 'Weight']
    assert (edges['Weight'] == 1).all()
