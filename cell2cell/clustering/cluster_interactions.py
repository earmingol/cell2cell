# -*- coding: utf-8 -*-

from __future__ import absolute_import

import community
import numpy as np
import pandas as pd

from cell2cell.core import networks
from cell2cell.extras import parallel_computing
from contextlib import closing
from multiprocessing import Pool


def clustering_interactions(interaction_elements, algorithm='louvain', method='raw', n_jobs=1, verbose=True):

    clustering = dict()
    clustering['algorithm'] = algorithm
    clustering['method'] = method
    clustering['raw_clusters'] = []


    if method == 'average':
        clustering['final_cci_matrix'] = compute_average(interaction_elements)
        if algorithm == 'louvain':
            clustering.update(community_detection(cci_matrix=clustering['final_cci_matrix'],
                                                  verbose=verbose))
        elif algorithm == 'hierarchical_louvain':
            pass
        else:
            raise ValueError("Clustering algorithm not implemented yet. Specify an implemented one")

    elif method == 'raw':
        # Parallel computing
        agents = parallel_computing.agents_number(n_jobs)
        chunksize = 1

        with closing(Pool(processes=agents)) as pool:
            if algorithm == 'louvain':
                inputs = [{'interaction_elements' : element, 'verbose' : verbose} for element in interaction_elements]
                clustering['raw_clusters'] = pool.map(parallel_computing.parallel_community_detection, inputs, chunksize)
                clustering['final_cci_matrix'] = compute_clustering_ratio(interaction_elements=interaction_elements,
                                                                          raw_clusters=clustering['raw_clusters'])
                clustering.update(community_detection(cci_matrix=clustering['final_cci_matrix'],
                                                      verbose=verbose))
            elif algorithm == 'hierarchical_louvain':
                pass
            else:
                raise ValueError("Clustering algorithm not implemented yet. Specify an implemented one")

    else:
        raise ValueError("Specificy a correct method. It could be 'average' or 'raw'.")


    return clustering

# Community algorithms
def community_detection(cci_matrix, randomize = True, verbose=True):
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
                                        package='networkx')

    # Compute the best partition
    partition = community.best_partition(G, randomize = randomize)

    modularity = np.round(community.modularity(partition, G),4)

    records = []
    for cell, cluster in partition.items():
        records.append((cell, cluster))

    results = dict()
    results['clusters'] = pd.DataFrame.from_records(records, columns=['Cell', 'Cluster'])
    results['modularity'] = modularity
    return results


def community_walktrap(cci_matrix):
    G = networks.network_from_adjacency(cci_matrix,
                                        package='igraph')

    hier_community = G.community_walktrap(weights='weight',
                                          steps=20)

    return hier_community


def compute_average(interaction_elements):
    matrices = [element['cci_matrix'].values for element in interaction_elements]
    mean_matrix = np.nanmean(matrices, axis=0)
    mean_matrix = pd.DataFrame(mean_matrix,
                               index=interaction_elements[0]['cci_matrix'].index,
                               columns=interaction_elements[0]['cci_matrix'].columns)
    mean_matrix = mean_matrix.fillna(0)
    return mean_matrix


def compute_clustering_ratio(interaction_elements, raw_clusters):
    '''
    This function computes the ratio of times that each pair of cells that was clustered together.
    @ Erick, PARALLELIZE this function
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
            pair_clustering.loc[clustered_cells, clustered_cells] = old_pair + np.ones(old_pair.shape) - np.identity(len(clustered_cells)) # Remove self interactions

    cci_matrix = pair_clustering / pair_participation

    return cci_matrix