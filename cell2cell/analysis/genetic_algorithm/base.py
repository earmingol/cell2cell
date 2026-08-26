# -*- coding: utf-8 -*-

'''The objective-function contract, and how to combine several of them.

The search in `search` knows nothing about what is being optimized. It asks an
**objective factory** for an objective bound to the current pool of candidate
ligand-receptor pairs, and then maximizes whatever that returns.

Two callables define the contract:

.. code-block:: text

    ObjectiveFactory : pool (DataFrame)     -> Objective
    Objective        : masks (n, n_pool)    -> fitness (n,)

The factory exists because of the successive runs: each one restricts the pool to
the pairs the previous run kept, so anything precomputed from the pool has to be
rebuilt against the smaller one. It is called once per run.

Requirements on a custom objective, each of which fails silently otherwise:

- **Higher is better.** The search maximizes.
- **Deterministic given the mask**, or `random_state` stops meaning anything and
  the consensus across executions measures noise instead of agreement.
- **Precompute in the factory, not per call.** An objective that rebuilds an
  interaction space per individual is thousands of times slower; the factory is
  the place for anything that depends only on the pool.
- **Finite.** Return a real number for every mask, including the all-zero one.
'''

from __future__ import absolute_import

import numpy as np


def _mean(values, weights=None):
    return np.average(values, axis=0, weights=weights)


COMBINERS = {
    'mean': _mean,
    'median': lambda values, weights=None: np.median(values, axis=0),
    'min': lambda values, weights=None: np.min(values, axis=0),
    'max': lambda values, weights=None: np.max(values, axis=0),
}


class CombinedObjective:
    '''
    Optimizes one set of ligand-receptor pairs against several datasets at once.

    Rather than searching each dataset separately and reconciling the answers
    afterwards, this evaluates every candidate on all of them and combines the
    result, so the search is driven towards pairs that work everywhere. With
    *D* datasets of *K* cell types the same candidate must satisfy
    `D * K * (K - 1) / 2` constraints instead of `K * (K - 1) / 2`, which is what
    makes it far less prone to fitting one dataset's noise.

    Parameters
    ----------
    factories : list
        Objective factories, one per dataset. Usually
        `cell2cell.analysis.genetic_algorithm.CorrelationObjective` instances, one
        per slide or donor, each with its own expression and reference matrices.
        They must all be built against the **same** pool of pairs, since a single
        mask indexes all of them.

    combine : str or callable, default='mean'
        How to reduce the per-dataset fitness values. One of 'mean', 'median',
        'min' or 'max', or a callable taking `(values, weights)` where `values` has
        shape (datasets, n) and returning shape (n,).

    sd_penalty : float, default=0.0
        Subtracts `sd_penalty` times the standard deviation across datasets from
        the combined value, so a candidate that does well everywhere is preferred
        over one that does very well on some datasets and poorly on others. The
        default of 0.0 leaves the combination unpenalized.

        The standard deviation is the population one (`ddof=0`), so it is defined
        for a single dataset -- where it is 0, and the whole thing reduces exactly
        to the single-dataset case. Since correlations are bounded, the penalty is
        on the same scale as the value; 0.25 to 1.0 is a sensible range, and 1.0
        already trades a full point of mean correlation for one standard deviation
        of disagreement.

    weights : array-like, default=None
        Per-dataset weights for the mean, for instance by cell count or quality.
        Ignored by combiners other than 'mean'.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> per_slide = [c2c.analysis.CorrelationObjective(rnaseq_data=rna,
    ...                                                reference_distances=ref,
    ...                                                cutoff_setup=cutoff,
    ...                                                analysis_setup=setup)
    ...              for rna, ref in slides]
    >>> objective = c2c.analysis.CombinedObjective(per_slide, sd_penalty=0.5)
    >>> results = c2c.analysis.optimize_lr_pairs(ppi_data=lr_pairs, objective=objective)
    '''

    def __init__(self, factories, combine='mean', sd_penalty=0.0, weights=None):
        if len(factories) == 0:
            raise ValueError('`factories` must hold at least one objective factory')
        if sd_penalty < 0:
            raise ValueError('`sd_penalty` must not be negative')
        if isinstance(combine, str) and combine not in COMBINERS:
            raise ValueError("`combine` must be a callable or one of {}"
                             .format(sorted(COMBINERS)))
        if weights is not None and len(weights) != len(factories):
            raise ValueError('`weights` must have one value per factory')

        self.factories = list(factories)
        self.combine = combine
        self.sd_penalty = float(sd_penalty)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def __call__(self, pool):
        return _BoundCombinedObjective(self, [factory(pool) for factory in self.factories])


class _BoundCombinedObjective:
    '''Several bound objectives over one pool, evaluated together.'''

    def __init__(self, parent, objectives):
        self.parent = parent
        self.objectives = objectives

    def evaluate_components(self, masks):
        '''
        Fitness of each dataset separately, of shape (datasets, n).

        Useful on the winning mask after the search, to see whether a solution is
        uniformly good or carried by a subset of the datasets -- the combined value
        alone cannot distinguish those.
        '''
        return np.vstack([np.atleast_1d(objective(masks)) for objective in self.objectives])

    def __call__(self, masks):
        values = self.evaluate_components(masks)
        combine = self.parent.combine
        reduce_fn = COMBINERS[combine] if isinstance(combine, str) else combine
        fitness = np.asarray(reduce_fn(values, self.parent.weights), dtype=float)
        if self.parent.sd_penalty:
            fitness = fitness - self.parent.sd_penalty * np.std(values, axis=0, ddof=0)
        return fitness
