# -*- coding: utf-8 -*-

from __future__ import absolute_import

import community
import leidenalg
import numpy as np
import pandas as pd

from cell2cell.core import networks
from cell2cell.extras import parallel_computing
from contextlib import closing
from multiprocessing import Pool


def clustering_interactions(interaction_elements, algorithm='louvain', method='raw', seed=None, package='networkx', n_jobs=1, verbose=True):

    clustering = dict()
    clustering['algorithm'] = algorithm
    if len(interaction_elements) == 1:
        print("Clustering method automatically set to 'average' because number of iterations is 1.")
        clustering['method'] = 'average'
    else:
        clustering['method'] = method
    clustering['raw_clusters'] = []


    if clustering['method'] == 'average':
        clustering['final_cci_matrix'] = compute_average(interaction_elements)
        if algorithm == 'louvain':
            clustering.update(community_detection(cci_matrix=clustering['final_cci_matrix'],
                                                  seed=seed,
                                                  package=package,
                                                  verbose=verbose))
        elif algorithm == 'hierarchical_louvain':
            pass
        elif algorithm == 'leiden':
            clustering.update(leiden_community(cci_matrix=clustering['final_cci_matrix'],
                                               verbose=verbose))
        else:
            raise ValueError("Clustering algorithm not implemented yet. Specify an implemented one")

    elif clustering['method'] == 'raw':
        # Parallel computing
        agents = parallel_computing.agents_number(n_jobs)
        chunksize = 1

        with closing(Pool(processes=agents)) as pool:
            inputs = [{'interaction_elements' : element, 'seed' : seed, 'package' : package, 'verbose' : verbose} for element in interaction_elements]

            if algorithm == 'louvain':
                clustering['raw_clusters'] = pool.map(parallel_computing.parallel_community_detection, inputs,
                                                      chunksize)
                clustering['final_cci_matrix'] = compute_clustering_ratio(interaction_elements=interaction_elements,
                                                                          raw_clusters=clustering['raw_clusters'])
                clustering.update(community_detection(cci_matrix=clustering['final_cci_matrix'],
                                                      verbose=verbose))
            elif algorithm == 'hierarchical_louvain':
                pass
            elif algorithm == 'leiden':
                clustering['raw_clusters'] = pool.map(parallel_computing.parallel_leiden_community, inputs,
                                                      chunksize)
                clustering['final_cci_matrix'] = compute_clustering_ratio(interaction_elements=interaction_elements,
                                                                          raw_clusters=clustering['raw_clusters'])
                clustering.update(leiden_community(cci_matrix=clustering['final_cci_matrix'],
                                                   verbose=verbose))
            else:
                raise ValueError("Clustering algorithm not implemented yet. Specify an implemented one")
    else:
        raise ValueError("Specificy a correct method. It could be 'average' or 'raw'.")


    return clustering

# Community algorithms
def community_detection(cci_matrix, randomize = True, seed = None, package='networkx', verbose=True):
    '''Compute Community Detection clustering (or Louvain method, for more info see https://arxiv.org/abs/0803.0476) and
    then generate the compartments containing cells that belong to the same cluster.

    Parameters
    ----------
    cci_matrix : Pandas DataFrame
        CCI matrix from computing interactions.

    randomize : boolean (default = True)
        Randomize parameter for community.best_partition function of community library.

    Returns
    -------
    results : dict
        Dictionary that contains ...
    '''

    if verbose:
        print("Performing Community Detection")

    G = networks.network_from_adjacency(cci_matrix,
                                        package=package)

    if package == 'networkx':
        # Compute the best partition
        if seed is not None:
            randomize = False

        partition = community.best_partition(G, randomize=randomize, random_state=seed)

        modularity = np.round(community.modularity(partition, G), 4)

        records = []
        for cell, cluster in partition.items():
            records.append((cell, cluster))
    elif package == 'igraph':
        partition = G.community_multilevel(weights='weight')

        modularity = partition.modularity
        records = []
        for i, cluster in enumerate(partition.membership):
            cell = partition.graph.vs(i).get_attribute_values('label')[0]
            records.append((cell, cluster))
    else:
        raise ValueError("Package not implemented. Use 'networkx' or 'igraph'.")

    results = dict()
    results['clusters'] = pd.DataFrame.from_records(records, columns=['Cell', 'Cluster'])
    results['modularity'] = modularity
    return results


def leiden_community(cci_matrix, verbose = True):


    if verbose:
        print("Performing Leiden Algorithm for community.")
    G = networks.network_from_adjacency(cci_matrix,
                                        package='igraph')

    partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)

    records = []
    for i, cluster in enumerate(partition.membership):
        cell = partition._graph.vs(i).get_attribute_values('label')[0]
        records.append((cell, cluster))

    results = dict()
    results['clusters'] = pd.DataFrame.from_records(records, columns=['Cell', 'Cluster'])
    results['modularity'] = partition.modularity
    return results


def hierarchical_community(cci_matrix, algorithm='walktrap'):
    G = networks.network_from_adjacency(cci_matrix,
                                        package='igraph')
    if algorithm == 'walktrap':
        hier_community = G.community_walktrap(weights='weight',
                                              steps=100)
    elif algorithm == 'betweenness':
        hier_community = G.community_edge_betweenness(directed=False,
                                                      weights='weight')
    elif algorithm == 'community_fastgreedy':
        hier_community = G.community_fastgreedy(weights='weight')
    else:
        raise NotImplementedError('Algorithm {} is not implemented. Try "walktrap", "betweenness" or "community_fastgreedy".')
    return hier_community


# Algorithms for computing a final CCI matrix from all iterations.
def compute_average(interaction_elements):
    matrices = [element['cci_matrix'].values for element in interaction_elements]
    mean_matrix = np.nanmean(matrices, axis=0)
    mean_matrix = pd.DataFrame(mean_matrix,
                               index=interaction_elements[0]['cci_matrix'].index,
                               columns=interaction_elements[0]['cci_matrix'].columns)
    mean_matrix = mean_matrix.fillna(0)
    # Remove self interactions
    #np.fill_diagonal(mean_matrix.values, 0)
    return mean_matrix


def compute_clustering_ratio(interaction_elements, raw_clusters):
    '''
    This function computes the ratio of times that each pair of cells that was clustered together.
    '''

    # Initialize matrices to store data
    total_cells = list(interaction_elements[0]['cci_matrix'].index)
    pair_participation = pd.DataFrame(np.zeros((len(total_cells), len(total_cells))),
                                      columns=total_cells,
                                      index=total_cells)

    pair_clustering = pair_participation.copy()

    # Compute ratios
    for i, element in enumerate(interaction_elements):
        included_cells = element['cells']

        # Update Total Participation
        old_values = pair_participation.loc[included_cells, included_cells].values
        pair_participation.loc[included_cells, included_cells] = old_values + np.ones(old_values.shape)

        # Update Pairs
        cluster_df = raw_clusters[i]['clusters']

        for cluster in cluster_df['Cluster'].unique():
            clustered_cells = cluster_df['Cell'].loc[cluster_df['Cluster'] == cluster].values
            old_pair = pair_clustering.loc[clustered_cells, clustered_cells].values
            pair_clustering.loc[clustered_cells, clustered_cells] = old_pair + np.ones(old_pair.shape) - np.identity(len(clustered_cells)) # Remove self interactions resulting from parallel clustering

    cci_matrix = pair_clustering / pair_participation
    cci_matrix = cci_matrix.replace(np.inf, np.nan)
    cci_matrix = cci_matrix.fillna(0)
    return cci_matrix