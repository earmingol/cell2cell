# -*- coding: utf-8 -*-

from __future__ import absolute_import

from multiprocessing import cpu_count


# GENERAL
def agents_number(n_jobs):
    '''
    This function computes the number of available agents based on the number of jobs provided.

    '''
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
    return agents


# CORE FUNCTIONS
def parallel_spatial_ccis(inputs):
    '''
    Parallel computing in cell2cell2.experiments.SpatialSingleCellExperiment
    '''
    # TODO: Implement this for enabling spatial analysis and compute interactions in parallel

    # from cell2cell.core import spatial_operation
    #results = spatial_operation()

    # return results
    pass