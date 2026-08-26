# -*- coding: utf-8 -*-

'''The genetic-algorithm search itself.

The only module here tied to `pygad`. Everything about *what* is being
optimized lives in `objectives`; this drives the search and the successive
runs that shrink the candidate pool.
'''

from __future__ import absolute_import

from types import ModuleType

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats

from cell2cell.core.interaction_space import InteractionSpace
from cell2cell.core.prepared_scorer import (PreparedCCIScorer, LINEAR_CCI_SCORES,
                                            UNBOUNDED_CCI_SCORES)
from cell2cell.preprocessing.ppi import (bidirectional_ppi_for_cci, bidirectional_index,
                                         remove_ppi_bidirectionality)
from cell2cell.analysis.genetic_algorithm.objectives import CorrelationObjective
from cell2cell.analysis.genetic_algorithm.consensus import (lr_selection_frequency,
                                                            lr_cooccurrence,
                                                            consensus_from_cooccurrence,
                                                            consensus_from_frequency)


_bidirectional_index = bidirectional_index


def _check_if_pygad() -> ModuleType:
    try:
        import pygad

    except Exception:
        raise ImportError('pygad is not installed. Please install it with: '
                          'pip install pygad'
                          )
    return pygad


def _reference_distance_matrix(interaction_space, ppi_score, cells):
    '''Distance matrix through the unmodified `InteractionSpace` code path.'''
    interaction_space.ppi_data['score'] = np.asarray(ppi_score, dtype=float)
    interaction_space.interaction_elements['ppi_score'] = interaction_space.ppi_data['score'].values
    interaction_space.compute_pairwise_cci_scores(use_ppi_score=True, verbose=False)
    return interaction_space.distance_matrix.loc[cells, cells]


def _correlation(distance_vector, reference_vector, method='spearman'):
    if method == 'spearman':
        corr = scipy.stats.spearmanr(distance_vector, reference_vector)[0]
    elif method == 'pearson':
        corr = scipy.stats.pearsonr(distance_vector, reference_vector)[0]
    else:
        raise ValueError("`method` must be either 'spearman' or 'pearson'")
    return abs(np.nan_to_num(corr))


def _resolve_objective(objective, rnaseq_data, reference_distances, cutoff_setup,
                       analysis_setup, **kwargs):
    '''Returns the objective factory to use, building the default when none is given.'''
    supplied = [rnaseq_data, reference_distances, cutoff_setup, analysis_setup]
    if objective is not None:
        if any(argument is not None for argument in supplied):
            raise ValueError(
                'Pass either `objective` or the data it would be built from '
                '(`rnaseq_data`, `reference_distances`, `cutoff_setup`, `analysis_setup`), '
                'not both -- otherwise it is ambiguous which one is in effect.')
        return objective
    if any(argument is None for argument in supplied):
        raise ValueError(
            'Without an `objective`, all of `rnaseq_data`, `reference_distances`, '
            '`cutoff_setup` and `analysis_setup` are required.')
    return CorrelationObjective(rnaseq_data=rnaseq_data,
                                reference_distances=reference_distances,
                                cutoff_setup=cutoff_setup,
                                analysis_setup=analysis_setup,
                                **kwargs)


def _optimize_once(rnaseq_data=None, ppi_data=None, reference_distances=None,
                    cutoff_setup=None, analysis_setup=None, objective=None,
                    included_cells=None, population_size=200, generations=200, runs=None,
                    inc_percentage=0.025, max_runs=100, correlation='spearman', signed=False,
                    mutation_probability=0.05, keep_elitism=1, random_state=None,
                    interaction_columns=('A', 'B'), complex_sep=None, complex_agg_method='min',
                    fast=True, validate_fast=True, max_memory_mb=512, deduplicate=True,
                    verbose=False):
    '''
    Selects the subset of ligand-receptor pairs whose cell-cell interaction scores
    best reproduce a reference distance between cells, using a genetic algorithm.

    Each individual is a binary vector with one entry per ligand-receptor pair,
    indicating whether it is included. The objective function is the absolute
    Spearman correlation between the resulting CCI distance matrix and
    `reference_distances`, as in Armingol et al. (2022) on the whole body of *C. elegans*.

    The search is repeated in successive runs: each run keeps only the pairs that
    the previous one selected, so the set shrinks until the objective stops
    improving by at least `inc_percentage`.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression matrix, with genes as rows and cells as columns.

    ppi_data : pandas.DataFrame
        List of ligand-receptor pairs. A 'score' column is added if missing.

    reference_distances : pandas.DataFrame
        Square, symmetric matrix of reference distances between cells, for example
        physical distances. Rows and columns are cell names.

    cutoff_setup : dict
        Cutoff setup, as in `cell2cell.analysis.initialize_interaction_space`.

    analysis_setup : dict
        Analysis setup with the keys 'communication_score', 'cci_score' and
        'cci_type', as in `cell2cell.analysis.initialize_interaction_space`.
        `cci_type` must be 'undirected', since the objective compares the
        condensed form of a symmetric distance matrix.

    included_cells : list, default=None
        Cells to consider. If None, the cells present in both `rnaseq_data` and
        `reference_distances` are used.

    population_size : int, default=200
        Number of individuals per generation.

    generations : int, default=200
        Number of generations per run.

    runs : int, default=None
        Number of runs. If None, runs continue until the objective improves by
        less than `inc_percentage` with respect to the previous run, or until
        `max_runs` is reached.

    inc_percentage : float, default=0.025
        Minimum relative improvement of the objective for another run to start.
        Only used when `runs` is None.

    max_runs : int, default=100
        Upper bound on the number of runs when `runs` is None.

    correlation : str, default='spearman'
        Correlation between the CCI distances and the reference distances, either
        'spearman' or 'pearson'.

    mutation_probability : float, default=0.05
        Probability of flipping each gene, equivalent to the flip mutator of the
        original implementation.

    keep_elitism : int, default=1
        Number of best individuals carried over to the next generation.

    random_state : int, default=None
        Seed for reproducibility.

    interaction_columns : tuple, default=('A', 'B')
        Columns of `ppi_data` holding the ligands and the receptors.

    complex_sep : str, default=None
        Separator of the subunits of a protein complex, if any.

    complex_agg_method : str, default='min'
        Method to aggregate the expression of the subunits of a complex.

    fast : boolean, default=True
        Whether to evaluate the objective with `PreparedCCIScorer`, which is
        equivalent but vectorized. If False, every individual is evaluated by
        rebuilding the CCI scores through `InteractionSpace`, exactly as the
        original implementation did.

    validate_fast : boolean, default=True
        Whether to check the vectorized objective against the reference one on a
        few random individuals before starting. Cheap, and it catches the case
        where the two paths would disagree.

    max_memory_mb : float, default=512
        Memory budget for the precomputed ligand-receptor outer products.

    deduplicate : boolean, default=True
        Whether to collapse each interaction and its reciprocal into a single row
        with `remove_ppi_bidirectionality` before the search. This is required for
        a pair to map onto a fixed set of bidirectional rows: `bidirectional_ppi_for_cci`
        drops duplicates on (A, B, score), so if both directions are present as
        separate rows, the number of bidirectional rows depends on the candidate
        solution. Pairs loaded with `cell2cell.io.load_ppi` are already deduplicated
        by `preprocess_ppi_data`, so this is a no-op for them; it matters when the
        table comes from somewhere else. Turn it off only if the input is already
        deduplicated -- the ambiguous case is rejected either way.
        Note that the returned masks are then indexed against the deduplicated
        table, which is also what 'best_ppi_data' contains.

    verbose : boolean, default=False
        Whether to print the progress of each run.

    Returns
    -------
    results : dict
        Dictionary with one entry per run, keyed 'run1', 'run2', ..., each with:

        - 'obj_fn' : the objective function of the best individual.
        - 'ppi_data' : list of 0/1, one per row of the original `ppi_data`,
          indicating the pairs selected in that run.
        - 'drop_fraction' : fraction of the pairs available to that run that were
          dropped.
        - 'n_selected' : number of pairs selected.

        The dictionary also holds 'best_run', 'best_obj_fn' and 'best_ppi_data',
        the last being a copy of `ppi_data` restricted to the selected pairs.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> results = c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq,
    ...                                          ppi_data=lr_pairs,
    ...                                          reference_distances=physical_distances,
    ...                                          cutoff_setup={'type': 'constant_value',
    ...                                                        'parameter': 10},
    ...                                          analysis_setup={'communication_score': 'expression_thresholding',
    ...                                                          'cci_score': 'bray_curtis',
    ...                                                          'cci_type': 'undirected'},
    ...                                          random_state=888)
    >>> selected = results['best_ppi_data']
    '''
    pygad = _check_if_pygad()

    if ppi_data is None:
        raise ValueError('`ppi_data` is required')
    objective = _resolve_objective(objective, rnaseq_data, reference_distances,
                                   cutoff_setup, analysis_setup,
                                   included_cells=included_cells,
                                   correlation=correlation, signed=signed, fast=fast,
                                   validate_fast=validate_fast,
                                   max_memory_mb=max_memory_mb,
                                   interaction_columns=interaction_columns,
                                   complex_sep=complex_sep,
                                   complex_agg_method=complex_agg_method,
                                   verbose=verbose)

    if deduplicate:
        # Required for the pair-to-bidirectional-row mapping to be well defined; see
        # `preprocessing.ppi.bidirectional_index`.
        ppi_data = remove_ppi_bidirectionality(ppi_data=ppi_data,
                                               interaction_columns=interaction_columns,
                                               verbose=verbose)
        ppi_data = ppi_data.drop_duplicates(subset=list(interaction_columns))
        ppi_data = ppi_data.reset_index(drop=True)

    theta_ppi_data = ppi_data.copy()
    if 'score' not in theta_ppi_data.columns:
        theta_ppi_data = theta_ppi_data.assign(score=1.0)

    prot_a, prot_b = interaction_columns
    results = dict()
    run = 1
    previous_obj = None

    while True:
        if runs is None:
            if run > max_runs:
                break
        elif run > runs:
            break

        # Each run searches only among the pairs the previous run kept
        theta_ppi_data = theta_ppi_data.loc[theta_ppi_data['score'] == 1].reset_index(drop=True)
        n_ppi = len(theta_ppi_data)
        if n_ppi == 0:
            break

        bound = objective(theta_ppi_data)

        def fitness_func(ga_instance, solution, solution_idx):
            return float(bound(np.atleast_2d(solution))[0])

        ga = pygad.GA(num_generations=generations,
                    num_parents_mating=max(2, population_size // 2),
                    fitness_func=fitness_func,
                    sol_per_pop=population_size,
                    num_genes=n_ppi,
                    gene_type=int,
                    init_range_low=0,
                    init_range_high=2,
                    gene_space=[0, 1],
                    parent_selection_type='tournament',
                    keep_elitism=keep_elitism,
                    mutation_type='random',
                    mutation_probability=mutation_probability,
                    random_seed=random_state if random_state is None else random_state + run,
                    suppress_warnings=True,
                    )
        # Evaluate the whole generation at once where the objective allows it
        ga.fitness_batch_size = population_size

        def batch_fitness(ga_instance, solutions, solutions_indices):
            return list(bound(np.atleast_2d(solutions)))

        ga.fitness_func = batch_fitness

        ga.run()

        best_solution, best_fitness, _ = ga.best_solution()
        best = np.asarray(best_solution, dtype=int)

        theta_ppi_data['score'] = best.astype(float)
        drop_fraction = 1.0 - best.sum() / len(best)

        # Map the selection back onto the rows of the original ppi_data
        selected = theta_ppi_data.loc[theta_ppi_data['score'] == 1, [prot_a, prot_b]]
        selected_pairs = set(map(tuple, selected.values))
        mask = [1 if tuple(row) in selected_pairs else 0
                for row in ppi_data[[prot_a, prot_b]].values]

        results['run{}'.format(run)] = {'obj_fn': float(best_fitness),
                                      'ppi_data': mask,
                                      'drop_fraction': float(drop_fraction),
                                      'n_selected': int(best.sum()),
                                      }
        if verbose:
            print('Run {}: objective {:.4f}, {} of {} pairs kept'
                  .format(run, best_fitness, int(best.sum()), len(best)))

        if runs is None and previous_obj is not None:
            if (best_fitness - previous_obj) / previous_obj < inc_percentage:
                run += 1
                break
        previous_obj = best_fitness
        run += 1

    if not results:
        raise RuntimeError('The genetic algorithm produced no results')

    best_key = max(results, key=lambda k: results[k]['obj_fn'])
    best_mask = np.asarray(results[best_key]['ppi_data'], dtype=bool)
    results['best_run'] = best_key
    results['best_obj_fn'] = results[best_key]['obj_fn']
    results['best_ppi_data'] = ppi_data.loc[best_mask].reset_index(drop=True)
    return results

def optimize_lr_pairs(rnaseq_data=None, ppi_data=None, reference_distances=None,
                      cutoff_setup=None, analysis_setup=None, objective=None,
                      executions=1, random_state=None, consensus_method='cooccurrence',
                      n_clusters=2, cluster_selection='cooccurrence', min_frequency=0.0,
                      frequency_percentile=90, verbose=False, **kwargs):
    '''
    Selects ligand-receptor pairs whose cell-cell interaction scores best reproduce a
    reference distance between cells, using a genetic algorithm.

    A genetic algorithm converges to a *local* optimum, so a single execution is not
    conclusive: different seeds settle on different, largely overlapping sets of
    pairs. With `executions > 1` the search is repeated independently and the results
    are integrated the way the reference analysis did -- by how often each pair is
    selected, and by which pairs are selected *together* -- which separates the
    reproducible core from the noise of any one execution.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression matrix, with genes as rows and cells as columns.

    ppi_data : pandas.DataFrame
        List of ligand-receptor pairs. A 'score' column is added if missing.

    reference_distances : pandas.DataFrame
        Square, symmetric matrix of reference distances between cells, for example
        physical distances. Rows and columns are cell names.

    cutoff_setup : dict
        Cutoff setup, as in `cell2cell.analysis.initialize_interaction_space`.

    analysis_setup : dict
        Analysis setup with the keys 'communication_score', 'cci_score' and
        'cci_type'. `cci_type` must be 'undirected'.

    executions : int, default=1
        Number of independent runs of the genetic algorithm. Each uses a different
        seed derived from `random_state`. With more than one, the consensus outputs
        described below are added to the result. **Around 30 or more is needed for
        the co-occurrence clustering to be stable**; the reference analysis used
        about a hundred. The frequency route tolerates fewer.

    random_state : int, default=None
        Seed. Execution *i* uses `random_state + i`, so the whole set is reproducible.

    consensus_method : str, default='cooccurrence'
        How to integrate the executions when there is more than one.

        - 'cooccurrence' : cluster the pairs by how often they are selected
            *together* and keep one cluster. This is what the reference analysis
            did, and it is able to tell apart groups of pairs that are each
            self-consistent.
        - 'frequency' : keep the pairs selected most often, above
            `frequency_percentile`. Simpler, and blind to which pairs go together.

    n_clusters : int, default=2
        Number of clusters to cut the co-occurrence dendrogram into.

    cluster_selection : str, default='cooccurrence'
        Which cluster to keep: 'cooccurrence' for the one whose members co-occur
        most with each other, or 'smallest' for the fewest members, which is
        literally what the reference notebook did. See `consensus_from_cooccurrence`.

    min_frequency : float, default=0.0
        Drop pairs selected in fewer than this fraction of executions before building
        the co-occurrence clusters, removing pairs that co-occur perfectly only
        because they were each chosen once, in the same execution. Note the
        co-occurrence route needs roughly 30 or more executions to be stable; with
        fewer, prefer `consensus_method='frequency'`. See
        `consensus_from_cooccurrence`.

    frequency_percentile : float, default=90
        Cutoff percentile when `consensus_method='frequency'`.

    verbose : boolean, default=False
        Whether to print the progress of each execution.

    **kwargs
        Passed to each individual search: `population_size`, `generations`, `runs`,
        `inc_percentage`, `max_runs`, `correlation`, `mutation_probability`,
        `keep_elitism`, `included_cells`, `interaction_columns`, `complex_sep`,
        `complex_agg_method`, `fast`, `validate_fast`, `max_memory_mb` and
        `deduplicate`. See `_optimize_once` for their meaning.

    Returns
    -------
    results : dict
        With a single execution, the result of that search: one entry per run
        ('run1', 'run2', ...) with 'obj_fn', 'ppi_data' (a 0/1 mask over the rows of
        `ppi_data`), 'drop_fraction' and 'n_selected', plus 'best_run',
        'best_obj_fn' and 'best_ppi_data'.

        With several executions, the same keys report the single best execution, and
        these are added:

        - 'executions' : the full result of each execution, keyed 'execution1', ...
        - 'selection_masks' : binary array of shape (executions, LR pairs), the mask
          each execution converged to.
        - 'selection_frequency' : dataframe of the pairs with the fraction of
          executions that selected each one.
        - 'cooccurrence' : Jaccard co-occurrence between pairs across executions.
        - 'consensus_ppi_data' : the consensus selection -- with the default method,
          the pairs of the co-occurrence cluster whose members are most consistently
          chosen together. **This is the recommended output.**
        - 'consensus_clusters' : every cluster, so the others can be inspected.
        - 'consensus_cluster_scores' : mean intra-cluster co-occurrence per cluster.

        With `consensus_method='frequency'`, 'consensus_threshold' holds the
        frequency cutoff instead of the cluster keys.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> results = c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq,
    ...                                          ppi_data=lr_pairs,
    ...                                          reference_distances=physical_distances,
    ...                                          cutoff_setup={'type': 'constant_value',
    ...                                                        'parameter': 10},
    ...                                          analysis_setup={'communication_score': 'expression_thresholding',
    ...                                                          'cci_score': 'bray_curtis',
    ...                                                          'cci_type': 'undirected'},
    ...                                          executions=20, random_state=888)
    >>> results['consensus_ppi_data']
    '''
    if executions < 1:
        raise ValueError('`executions` must be at least 1')

    common = dict(rnaseq_data=rnaseq_data, ppi_data=ppi_data,
                  reference_distances=reference_distances, cutoff_setup=cutoff_setup,
                  analysis_setup=analysis_setup, objective=objective,
                  verbose=verbose, **kwargs)

    if executions == 1:
        return _optimize_once(random_state=random_state, **common)

    interaction_columns = kwargs.get('interaction_columns', ('A', 'B'))
    prot_a, prot_b = interaction_columns

    # The pool the masks are indexed against, matching what each execution searches
    pool = ppi_data
    if kwargs.get('deduplicate', True):
        pool = remove_ppi_bidirectionality(ppi_data=ppi_data,
                                           interaction_columns=interaction_columns,
                                           verbose=False)
        pool = pool.drop_duplicates(subset=list(interaction_columns)).reset_index(drop=True)

    all_executions, masks = {}, []
    for i in range(executions):
        seed = None if random_state is None else random_state + i
        result = _optimize_once(random_state=seed, **common)
        all_executions['execution{}'.format(i + 1)] = result
        # The pairs that execution converged to, i.e. its last run
        last = max((k for k in result if k.startswith('run')),
                   key=lambda k: int(k[3:]))
        masks.append(result[last]['ppi_data'])
        if verbose:
            print('Execution {}: best objective {:.4f}, {} pairs'
                  .format(i + 1, result['best_obj_fn'], sum(result[last]['ppi_data'])))

    masks = np.asarray(masks, dtype=int)
    labels = ['{}^{}'.format(a, b) for a, b in pool[[prot_a, prot_b]].values]

    frequency = pd.DataFrame({prot_a: pool[prot_a].values, prot_b: pool[prot_b].values,
                              'frequency': lr_selection_frequency(masks)})
    cooccurrence = lr_cooccurrence(masks, labels=labels)

    best_key = max(all_executions, key=lambda k: all_executions[k]['best_obj_fn'])
    results = dict(all_executions[best_key])
    results['executions'] = all_executions
    results['best_execution'] = best_key
    results['selection_masks'] = masks
    results['selection_frequency'] = frequency.sort_values('frequency', ascending=False)
    results['cooccurrence'] = cooccurrence

    if consensus_method == 'frequency':
        mask, threshold = consensus_from_frequency(frequency['frequency'].values,
                                                   percentile=frequency_percentile)
        results['consensus_ppi_data'] = pool.loc[mask].reset_index(drop=True)
        results['consensus_threshold'] = threshold
        if verbose:
            print('Frequency consensus: {} pairs above {:.3f}'
                  .format(int(mask.sum()), threshold))
    elif consensus_method == 'cooccurrence':
        try:
            selected, clusters, scores = consensus_from_cooccurrence(
                cooccurrence, n_clusters=n_clusters, select=cluster_selection,
                frequency=frequency['frequency'].values, min_frequency=min_frequency)
            chosen = set(selected)
            results['consensus_ppi_data'] = pool.loc[[l in chosen for l in labels]].reset_index(drop=True)
            results['consensus_clusters'] = clusters
            results['consensus_cluster_scores'] = scores
            if verbose:
                print('Co-occurrence consensus: {} pairs, cluster sizes {}, mean co-occurrence {}'
                      .format(len(chosen), {k: len(v) for k, v in clusters.items()},
                              {k: round(v, 3) for k, v in scores.items()}))
        except ValueError as error:
            if verbose:
                print('No consensus could be built: {}'.format(error))
            results['consensus_ppi_data'] = None
            results['consensus_clusters'] = None
            results['consensus_cluster_scores'] = None
    else:
        raise ValueError("`consensus_method` must be either 'cooccurrence' or 'frequency'")
    return results


def _as_symmetric(matrix):
    '''
    Validates a reference distance matrix and makes it exactly symmetric.

    `check_symmetry` compares with exact equality, which a matrix produced by a
    distance function often fails by a few ULP. Rather than rejecting those, they
    are averaged with their transpose; genuinely asymmetric input still raises.
    '''
    values = np.asarray(matrix.values, dtype=float)
    if values.shape[0] != values.shape[1]:
        raise ValueError('`reference_distances` must be a square matrix')
    if list(matrix.index) != list(matrix.columns):
        raise ValueError('`reference_distances` must have the same cells as rows and columns')
    if not np.allclose(values, values.T, rtol=1e-8, atol=1e-8, equal_nan=True):
        raise ValueError('`reference_distances` must be a symmetric matrix')
    return pd.DataFrame((values + values.T) / 2.0, index=matrix.index, columns=matrix.columns)
