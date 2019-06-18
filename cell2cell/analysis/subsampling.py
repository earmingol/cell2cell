# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd
import random

from cell2cell.core import InteractionSpace
from cell2cell.extras import parallel_computing
from contextlib import closing
from multiprocessing import Pool

class SubsamplingSpace:

    def __init__(self, rnaseq_data, ppi_dict, interaction_type, gene_cutoffs, subsampling_percentage=0.8,
                 iterations=1000, n_jobs=1, verbose=True):

        if verbose:
            print("Computing interactions by subsampling a"
                  " {}% of cells in each of the {} iterations".format(int(100*subsampling_percentage), iterations))

        self.cell_ids = list(rnaseq_data.columns)

        data = np.empty((len(self.cell_ids), len(self.cell_ids)))
        data[:] = np.nan
        self.cci_matrix_template = pd.DataFrame(data,
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


    def subsampling_interactions(self, rnaseq_data, ppi_dict, interaction_type, gene_cutoffs,
                                 subsampling_percentage=0.8, iterations=1000, n_jobs=1, verbose=True):
        '''
        This function performs the subsampling method by generating a list of cells to consider in each iteration.
        Then, for each list of cells a InteractionSpace is generated to perform the cell2cell analysis and return the
        respective CCI Matrix and clusters.
        '''
        # Last position for subsampling when shuffling
        last_item = int(len(self.cell_ids) * subsampling_percentage)

        inputs = {'cells' : self.cell_ids,
                  'list_end' : last_item,
                  'rnaseq' : rnaseq_data,
                  'ppi' : ppi_dict,
                  'interaction_type' : interaction_type,
                  'cutoffs' : gene_cutoffs,
                  'cci_matrix' : self.cci_matrix_template,
                  'verbose' :verbose
                  }

        inputs = [inputs]*iterations


        # Parallel computing
        agents = parallel_computing.agents_number(n_jobs)
        chunksize = 1
        with closing(Pool(processes=agents)) as pool:
            results = pool.map(parallel_computing.parallel_subsampling_interactions, inputs, chunksize)
        return results


def subsampling_operation(cell_ids, last_item, rnaseq_data, ppi_dict, interaction_type, gene_cutoffs,
                          cci_matrix_template = None, verbose=True):
    '''
    Functional unit to perform parallel computing in SubsamplingSpace
    '''
    cell_ids_ = cell_ids.copy()
    random.shuffle(cell_ids_)
    included_cells = cell_ids_[:last_item]

    interaction_space = InteractionSpace(rnaseq_data[included_cells],
                                         ppi_dict,
                                         interaction_type,
                                         gene_cutoffs,
                                         cci_matrix_template=cci_matrix_template,
                                         verbose=verbose)

    interaction_space.compute_pairwise_interactions(verbose=verbose)

    iteration_elements = dict()
    iteration_elements['cells'] = included_cells
    iteration_elements['cci_matrix'] = interaction_space.interaction_elements['cci_matrix']
    return iteration_elements