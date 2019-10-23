# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd
import random

from cell2cell.clustering import compute_avg_cci_matrix
from cell2cell.core import InteractionSpace
from cell2cell.utils import parallel_computing
from contextlib import closing
from multiprocessing import Pool


class SubsamplingSpace:

    def __init__(self, rnaseq_data, ppi_data, gene_cutoffs, score_type='binary', score_metric='bray_curtis',
                 subsampling_percentage=1.0, iterations=1, initial_seed=None, n_jobs=1, verbose=True):

        if verbose:
            print("Computing interactions by sub-sampling a"
                  " {}% of cells in each of the {} iterations".format(int(100*subsampling_percentage), iterations))

        self.cell_ids = list(rnaseq_data.columns)

        data = np.empty((len(self.cell_ids), len(self.cell_ids)))
        data[:] = np.nan
        self.cci_matrix_template = pd.DataFrame(data,
                                                columns=self.cell_ids,
                                                index=self.cell_ids)

        self.subsampled_interactions = self.subsampling_interactions(rnaseq_data=rnaseq_data,
                                                                     ppi_data=ppi_data,
                                                                     gene_cutoffs=gene_cutoffs,
                                                                     score_type=score_type,
                                                                     score_metric=score_metric,
                                                                     subsampling_percentage=subsampling_percentage,
                                                                     iterations=iterations,
                                                                     initial_seed=initial_seed,
                                                                     n_jobs=n_jobs,
                                                                     verbose=verbose)

        self.average_cci_matrix = compute_avg_cci_matrix(self)


    def subsampling_interactions(self, rnaseq_data, ppi_data, gene_cutoffs, score_type='binary',
                                 score_metric='bray_curtis', subsampling_percentage=1.0, iterations=1, initial_seed=None,
                                 n_jobs=1, verbose=True):
        '''
        This function performs the sub-sampling method by generating a list of cells to consider in each iteration.
        Then, for each list of cells a InteractionSpace is generated to perform the cell2cell analysis and return the
        respective CCI Matrix and clusters.
        '''
        # Last position for sub-sampling when shuffling
        last_item = int(len(self.cell_ids) * subsampling_percentage)
        if last_item == 0:
            last_item = 1

        inputs = {'cells' : self.cell_ids,
                  'list_end' : last_item,
                  'rnaseq' : rnaseq_data,
                  'ppi' : ppi_data,
                  'cutoffs' : gene_cutoffs,
                  'score_type' : score_type,
                  'score_metric' : score_metric,
                  'cci_matrix' : self.cci_matrix_template,
                  'verbose' :verbose
                  }

        inputs_list = []
        for i in range(iterations):
            inputs_ = inputs.copy()
            if initial_seed is not None:
                inputs_['seed'] = initial_seed + i
            else:
                inputs_['seed'] = None
            inputs_list.append(inputs_)

        # Parallel computing
        if n_jobs == 1:
            results = list(map(parallel_computing.parallel_subsampling_interactions, inputs_list))
        else:
            agents = parallel_computing.agents_number(n_jobs)
            chunksize = 1
            with closing(Pool(processes=agents)) as pool:
                results = pool.map(parallel_computing.parallel_subsampling_interactions, inputs_list, chunksize)
        return results


def subsampling_operation(cell_ids, last_item, rnaseq_data, ppi_data, gene_cutoffs, score_type='binary',
                          score_metric='bray_curtis', cci_matrix_template=None, seed=None, verbose=True):
    '''
    Functional unit to perform parallel computing in Sub-sampling Space
    '''
    cell_ids_ = cell_ids.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(cell_ids_)
    included_cells = cell_ids_[:last_item]

    interaction_space = InteractionSpace(rnaseq_data=rnaseq_data[included_cells],
                                         ppi_data=ppi_data,
                                         gene_cutoffs=gene_cutoffs,
                                         score_type=score_type,
                                         cci_matrix_template=cci_matrix_template,
                                         verbose=verbose)

    interaction_space.compute_pairwise_interactions(score_metric=score_metric,
                                                    verbose=verbose)

    interaction_elements = dict()
    interaction_elements['cells'] = included_cells
    interaction_elements['cci_matrix'] = interaction_space.interaction_elements['cci_matrix']
    interaction_elements['ppis'] = interaction_space.interaction_elements['ppis']
    return interaction_elements