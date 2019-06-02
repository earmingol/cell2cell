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

    if method == 'average':
        cci_matrix = compute_average(interaction_elements)
        if algorithm == 'louvain':
            clustering.update(community_detection(cci_matrix=cci_matrix,
                                                  verbose=verbose))
        elif algorithm == 'hierarchical_louvain':
            pass
        else:
            raise ValueError("Clustering algorithm not implemented yet. Specify an implemented one")

        clustering['raw_clusters'] = []
        clustering['final_cci_matrix'] = cci_matrix

    elif method == 'raw':
        # Parallel computing
        agents = parallel_computing.agents_number(n_jobs)
        chunksize = 1

        with closing(Pool(processes=agents)) as pool:
            if algorithm == 'louvain':
                inputs = [{'interaction_elements' : element, 'verbose' : verbose} for element in interaction_elements]
                clustering['raw_clusters'] = pool.map(parallel_computing.parallel_community_detection, inputs, chunksize)
            elif algorithm == 'hierarchical_louvain':
                pass
            else:
                raise ValueError("Clustering algorithm not implemented yet. Specify an implemented one")

        # Compute final clusters from individual iterations
        # Generate final CCI matrix
        # Store them into clustering dictionary
        #clustering['final_cci_matrix'] = cci_matrix_from_raw(clustering['raw'],
        #                                               algorithm=algorithm)
        #clustering.update(clustering_algorithm)
    else:
        raise ValueError("Specificy a correct method. It could be 'average' or 'raw'.")


    return clustering


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
    compartments : list
        List containing other lists, where each of these lists is a compartment that contain the name or id of each cell
        that belongs to it.
    '''

    if verbose:
        print("Performing Community Detection")

    G = networks.network_from_adjacency(cci_matrix)

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


def compute_average(interaction_elements):
    matrices = [element['cci_matrix'].values for element in interaction_elements]
    mean_matrix = np.nanmean(matrices, axis=0)
    mean_matrix = pd.DataFrame(mean_matrix,
                               index=interaction_elements[0]['cci_matrix'].index,
                               columns=interaction_elements[0]['cci_matrix'].columns)
    mean_matrix = mean_matrix.fillna(0)
    return mean_matrix