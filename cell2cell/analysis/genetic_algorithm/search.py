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
                                         remove_ppi_bidirectionality, deduplicate_ppi_pairs)
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
                          "pip install 'pygad>=3.0.0'"
                          )
    return pygad


def _deduplicate_pool(ppi_data, interaction_columns, duplicates='highest', verbose=False):
    '''One row per interaction: reciprocals collapsed, then pairs listed more than once.

    Both searches reduce the pool the same way, so the masks of a single search and
    those of several executions are indexed against the same table.
    '''
    ppi_data = remove_ppi_bidirectionality(ppi_data=ppi_data,
                                           interaction_columns=interaction_columns,
                                           verbose=verbose)
    return deduplicate_ppi_pairs(ppi_data, interaction_columns=interaction_columns,
                                 keep=duplicates, verbose=False)


_EPS = 1e-12


def _relative_improvement(best, previous):
    '''How much `best` improves on `previous`, relative to its magnitude.

    Dividing by `previous` itself would flip the sign of the comparison whenever the
    objective is negative -- an improvement from -0.50 to -0.40 would read as -0.20
    and look like a stall -- and would divide by zero when a run scores exactly 0.
    Objectives are only required to be "higher is better", so neither case is
    unusual: a signed correlation and a custom objective can both be negative.
    '''
    return (best - previous) / max(abs(previous), _EPS)


def _validate_ppi_score(ppi_data):
    '''Rejects interaction weights the objectives cannot make sense of.

    The 'score' column is the weight of each interaction, and is left untouched by
    the search -- the candidate selection is kept separately. Weights that are
    missing or negative have no meaning in the CCI scores, so they are refused here
    rather than propagating into a fitness value.
    '''
    if 'score' not in ppi_data.columns:
        return
    try:
        values = np.asarray(ppi_data['score'].values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("The 'score' column of `ppi_data` must be numeric, since it "
                         'is the weight of each interaction.')
    if not np.isfinite(values).all():
        # Infinities propagate into the CCI scores as inf, or as NaN through inf/inf
        raise ValueError("The 'score' column of `ppi_data` has values that are not "
                         'finite. It is the weight of each interaction, so every row '
                         'needs a real number.')
    if (values < 0.0).any():
        raise ValueError("The 'score' column of `ppi_data` has negative weights, which "
                         'the CCI scores are not defined for.')


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
                    duplicates='highest',
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
        List of ligand-receptor pairs. A 'score' column, if present, is the weight of
        each interaction: it is preserved on output and multiplied into the weights a
        candidate selection produces, so a pair contributes its own weight when
        selected and nothing when not. Missing means every pair weighs 1.

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
        Minimum relative improvement of the objective for another run to start,
        measured against the **magnitude** of the previous run's objective, so that a
        negative or zero objective is handled the same way a positive one is. Only
        used when `runs` is None.

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
        Whether to reduce the list to one row per interaction before the search, with
        `remove_ppi_bidirectionality` followed by `deduplicate_ppi_pairs`.

        The undirected CCI scores need the interactions in both directions, so the
        table is doubled internally. An interaction the input already lists in both
        directions would then appear twice over, and be weighted twice as heavily as
        the rest for no reason. A pair simply listed twice would do the same, and
        would take up two positions of the candidate set while being one interaction.
        Collapsing them first avoids both. Pairs loaded with `cell2cell.io.load_ppi`
        are already deduplicated by `preprocess_ppi_data`, so this is a no-op for them.

        Set it to False to search a list that deliberately repeats a pair, for
        instance the same interaction with the weight two databases give it. Each copy
        is then a position of its own in the candidate set, and the rows have to
        differ somewhere -- rows that repeat exactly cannot be selected independently
        and are rejected.

        Note that the returned masks are then indexed against the deduplicated
        table, which is also what 'best_ppi_data' contains.

    duplicates : str, default='highest'
        Which row to keep for a pair listed more than once, when `deduplicate` is
        True: 'highest' (the default) keeps the largest score, 'lowest' the smallest,
        'first' the one appearing first. See `deduplicate_ppi_pairs`.

    verbose : boolean, default=False
        Whether to print the progress of each run.

    Returns
    -------
    results : dict
        Dictionary with one entry per run, keyed 'run1', 'run2', ..., each with:

        - 'obj_fn' : the objective function of the best individual.
        - 'ppi_data' : list of 0/1, one per row of `results['pool']`, indicating
          the pairs selected in that run.
        - 'drop_fraction' : fraction of the pairs available to that run that were
          dropped.
        - 'n_selected' : number of pairs selected.

        The dictionary also holds 'best_run', 'best_obj_fn', 'best_ppi_data' (a copy
        of the pool restricted to the selected pairs) and 'pool', the deduplicated
        pairs the masks are indexed against.

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
        # Not a mechanical requirement -- the mapping onto bidirectional rows is exact
        # for any input. This is about not counting an interaction twice when the table
        # lists it in both directions, which the doubling would then triple.
        ppi_data = _deduplicate_pool(ppi_data, interaction_columns=interaction_columns,
                                     duplicates=duplicates, verbose=verbose)

    theta_ppi_data = ppi_data.copy()
    _validate_ppi_score(theta_ppi_data)

    # Which pairs the next run may choose from, kept apart from the 'score' column so
    # that an input weight is never mistaken for a selection. Every pair is available
    # to the first run.
    selection = np.ones(len(theta_ppi_data), dtype=bool)
    # Row of `ppi_data` each surviving candidate is, so a selection maps back by
    # position. Two rows can list the same pair when `deduplicate` is False, and they
    # are then separate candidates -- recovering them by their names would select both.
    positions = np.arange(len(theta_ppi_data))

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
        theta_ppi_data = theta_ppi_data.loc[selection].reset_index(drop=True)
        positions = positions[selection]
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

        selection = best.astype(bool)
        drop_fraction = 1.0 - best.sum() / len(best)

        # Map the selection back onto the rows of the original ppi_data, by position
        mask = np.zeros(len(ppi_data), dtype=int)
        mask[positions[selection]] = 1
        mask = mask.tolist()

        results['run{}'.format(run)] = {'obj_fn': float(best_fitness),
                                      'ppi_data': mask,
                                      'drop_fraction': float(drop_fraction),
                                      'n_selected': int(best.sum()),
                                      }
        if verbose:
            print('Run {}: objective {:.4f}, {} of {} pairs kept'
                  .format(run, best_fitness, int(best.sum()), len(best)))

        if runs is None and previous_obj is not None:
            if _relative_improvement(best_fitness, previous_obj) < inc_percentage:
                run += 1
                break
        previous_obj = best_fitness
        run += 1

    if not results:
        raise RuntimeError('The genetic algorithm produced no results')

    best_key = max(results, key=lambda k: results[k]['obj_fn'])
    best_mask = np.asarray(results[best_key]['ppi_data'], dtype=bool)
    # The frame the masks are indexed against, so a mask can be mapped back onto
    # pair annotations. Also present on the multi-execution result.
    results['pool'] = ppi_data
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
        List of ligand-receptor pairs. A 'score' column, if present, is the weight of
        each interaction: it is preserved on output and multiplied into the weights a
        candidate selection produces, so a pair contributes its own weight when
        selected and nothing when not. Missing means every pair weighs 1.

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
        `complex_agg_method`, `fast`, `validate_fast`, `max_memory_mb`,
        `deduplicate` and `duplicates`. See `_optimize_once` for their meaning.

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
        - 'pool' : the deduplicated pairs the masks are indexed against. **Columns of
          'selection_masks' follow this order**, so this is what to use when mapping
          a mask back onto pair annotations.
        - 'selection_masks' : binary array of shape (executions, LR pairs), the mask
          each execution converged to.
        - 'selection_frequency' : the pairs with the fraction of executions that
          selected each one, **sorted by frequency** for reading. Its index is the
          position in 'pool', so `.sort_index()` realigns it with 'selection_masks';
          pairing it positionally with the masks as returned would misalign them.
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
        pool = _deduplicate_pool(ppi_data, interaction_columns=interaction_columns,
                                 duplicates=kwargs.get('duplicates', 'highest'),
                                 verbose=False)

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
    # The position is part of the label so that two rows listing the same pair, which
    # `deduplicate=False` allows, stay distinguishable: the co-occurrence matrix is
    # indexed by these, and duplicated names would make it ambiguous.
    labels = ['{}^{}^{}'.format(a, b, position) for position, (a, b)
              in enumerate(pool[[prot_a, prot_b]].values)]

    frequency = pd.DataFrame({prot_a: pool[prot_a].values, prot_b: pool[prot_b].values,
                              'frequency': lr_selection_frequency(masks)})
    cooccurrence = lr_cooccurrence(masks, labels=labels)

    best_key = max(all_executions, key=lambda k: all_executions[k]['best_obj_fn'])
    results = dict(all_executions[best_key])
    results['executions'] = all_executions
    results['best_execution'] = best_key
    results['pool'] = pool
    results['selection_masks'] = masks
    # Sorted for reading, but the pool index is kept so it can be realigned with
    # `selection_masks` -- whose columns follow `pool` order -- via `.sort_index()`
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
            keep = np.array([label in chosen for label in labels])
            results['consensus_ppi_data'] = pool.loc[keep].reset_index(drop=True)
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
