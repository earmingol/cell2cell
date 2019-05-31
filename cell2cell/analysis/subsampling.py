# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd
import random

from cell2cell.core import InteractionSpace
#from cell2cell.clustering import clustering_interactions
from multiprocessing import Pool, cpu_count
from contextlib import closing

class SubsamplingSpace:

    def __init__(self, rnaseq_data, ppi_dict, interaction_type, gene_cutoffs, subsampling_percentage=0.8, iterations=1000, n_jobs=1, verbose=True):
        if verbose:
            print("Computing interactions by subsampling a {}% of cells in each of the {} iterations".format(int(100*subsampling_percentage), iterations))
        self.cell_ids = list(rnaseq_data.columns)

        self.cci_matrix_template = pd.DataFrame(np.zeros((len(self.cell_ids), len(self.cell_ids))),
                                                         columns=self.cell_ids,
                                                         index=self.cell_ids)

        self.subsampled_interactions = self.subsampling_interactions(rnaseq_data=rnaseq_data,
                                                                     ppi_dict=ppi_dict,
                                                                     interaction_type=interaction_type,
                                                                     gene_cutoffs=gene_cutoffs,
                                                                     subsampling_percentage=subsampling_percentage,
                                                                     iterations=iterations,
                                                                     n_jobs=n_jobs,
                                                                     verbose=verbose)


    def subsampling_interactions(self, rnaseq_data, ppi_dict, interaction_type, gene_cutoffs, subsampling_percentage=0.8, iterations=1000, n_jobs=1, verbose=True):
        '''
        This function performs the subsampling method by generating a list of cells to consider in each iteration.
        Then, for each list of cells a InteractionSpace is generated to perform the cell2cell anaysis and return the
        respective CCI Matrix and clusters.
        '''
        # Defining number of cores
        if n_jobs < 0:
            agents = cpu_count() + 1 + n_jobs
            if agents < 0:
                agents = 1
        elif n_jobs > cpu_count():
            agents = cpu_count()

        elif n_jobs == 0:
            agents = 1
        else:
            agents = n_jobs

        # Last position for subsampling when shuffling
        last_item = int(len(self.cell_ids) * subsampling_percentage)

        inputs = [self.cell_ids,
                  last_item,
                  rnaseq_data,
                  ppi_dict,
                  interaction_type,
                  gene_cutoffs,
                  self.cci_matrix_template,
                  verbose
                  ]

        inputs = [inputs]*iterations

        chunksize=1
        with closing(Pool(processes=agents)) as pool:
            results = pool.map(subsampling_unit, inputs, chunksize)
        return results


def subsampling_unit(inputs):
    '''
    Functional unit to perform parallel computing in SubsamplingSpace
    '''
    cell_ids = inputs[0]
    last_item = inputs[1]
    rnaseq_data = inputs[2]
    ppi_dict = inputs[3]
    interaction_type = inputs[4]
    gene_cutoffs = inputs[5]
    cci_matrix_template = inputs[6]
    verbose = inputs[7]

    random.shuffle(cell_ids)
    included_cells = cell_ids[:last_item]

    interaction_space = InteractionSpace(rnaseq_data[included_cells],
                                         ppi_dict,
                                         interaction_type,
                                         gene_cutoffs,
                                         cci_matrix_template=cci_matrix_template,
                                         verbose=verbose)

    interaction_space.compute_pairwise_interactions(verbose=verbose)

    # clustering
    # clusters = clustering_interactions(interaction_space.interaction_elements['cci_matrix'],
    #                                   method='louvain')

    iteration_elements = (interaction_space.interaction_elements['cci_matrix'])  # , clusters)
    return iteration_elements