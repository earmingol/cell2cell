# -*- coding: utf-8 -*-

from __future__ import absolute_import

from multiprocessing import cpu_count


# GENERAL
def agents_number(n_jobs):
    '''
    Computes the number of agents/cores/threads that the
    computer can really provide given a number of
    jobs/threads requested.

    Parameters
    ----------
    n_jobs : int
        Number of threads for parallelization.

    Returns
    -------
    agents : int
        Number of threads that the computer can really provide.
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
    Parallel computing in cell2cell2.analysis.pipelines.SpatialSingleCellInteractions
    '''
    # TODO: Implement this for enabling spatial analysis and compute interactions in parallel

    # from cell2cell.core import spatial_operation
    #results = spatial_operation()

    # return results
    pass