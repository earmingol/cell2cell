# -*- coding: utf-8 -*-

'''Tests for cell2cell.analysis.genetic_algorithm'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.analysis import genetic_algorithm as ga
from cell2cell.core.interaction_space import InteractionSpace
from cell2cell.preprocessing.ppi import bidirectional_ppi_for_cci, remove_ppi_bidirectionality

pygad = pytest.importorskip('pygad')


@pytest.fixture
def ga_inputs():
    '''Expression, LR pairs and a reference distance matrix over the same cells.'''
    n_cells = 10
    genes = ['G{}'.format(i) for i in range(150)]
    rnaseq = c2c.datasets.generate_random_rnaseq(size=n_cells, row_names=genes,
                                                 random_state=0, verbose=False)
    # Distinct ligand and receptor pools, so the pairs are not all self-interactions
    ppi = c2c.datasets.generate_random_ppi(max_size=40, interactors_A=genes[:75],
                                           interactors_B=genes[75:],
                                           random_state=0, verbose=False)
    rng = np.random.default_rng(0)
    coords = rng.random((n_cells, 2)) * 100
    from sklearn.metrics.pairwise import euclidean_distances
    reference = pd.DataFrame(euclidean_distances(coords, coords),
                             index=rnaseq.columns, columns=rnaseq.columns)
    return rnaseq, ppi, reference


@pytest.fixture
def setups():
    analysis_setup = {'communication_score': 'expression_thresholding',
                      'cci_score': 'bray_curtis',
                      'cci_type': 'undirected'}
    cutoff_setup = {'type': 'constant_value', 'parameter': 10}
    return analysis_setup, cutoff_setup


# ---------------------------------------------------------------------------------
# PreparedCCIScorer -- must reproduce InteractionSpace exactly
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize('communication_score,cci_score', [
    ('expression_thresholding', 'bray_curtis'),
    ('expression_thresholding', 'jaccard'),
    ('expression_thresholding', 'count'),
    ('expression_product', 'icellnet'),
    ('expression_product', 'bray_curtis'),
])
def test_prepared_scorer_matches_interaction_space(ga_inputs, communication_score, cci_score):
    '''The whole point of the vectorized scorer is that it changes nothing.'''
    rnaseq, ppi, _ = ga_inputs
    bi_ppi = bidirectional_ppi_for_cci(ppi, verbose=False)
    space = InteractionSpace(rnaseq_data=rnaseq, ppi_data=bi_ppi,
                             gene_cutoffs={'type': 'constant_value', 'parameter': 10},
                             communication_score=communication_score,
                             cci_score=cci_score, cci_type='undirected', verbose=False)
    scorer = ga.PreparedCCIScorer(space, cci_score=cci_score)
    source = ga._bidirectional_index(ppi, verbose=False)

    rng = np.random.default_rng(1)
    for _ in range(3):
        weights = rng.integers(0, 2, size=len(ppi)).astype(float)[source]

        space.ppi_data['score'] = weights
        space.compute_pairwise_cci_scores(use_ppi_score=True, verbose=False)
        expected_cci = space.interaction_elements['cci_matrix'].values.astype(float)
        expected_distance = space.distance_matrix.values.astype(float)

        got_cci = scorer.score_batch(weights[None, :])[0]
        got_distance = scorer.distance_batch(weights[None, :])[0]

        # Relative, because 'icellnet' is unbounded and BLAS sums in a different
        # order than the reference loop
        np.testing.assert_allclose(got_cci, expected_cci, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got_distance, expected_distance, rtol=1e-12, atol=1e-12)


def test_prepared_scorer_batch_matches_individual(ga_inputs):
    '''Scoring a population at once equals scoring its members one by one.'''
    rnaseq, ppi, _ = ga_inputs
    bi_ppi = bidirectional_ppi_for_cci(ppi, verbose=False)
    space = InteractionSpace(rnaseq_data=rnaseq, ppi_data=bi_ppi,
                             gene_cutoffs={'type': 'constant_value', 'parameter': 10},
                             communication_score='expression_thresholding',
                             cci_score='bray_curtis', cci_type='undirected', verbose=False)
    scorer = ga.PreparedCCIScorer(space)

    rng = np.random.default_rng(2)
    W = rng.integers(0, 2, size=(6, scorer.n_ppi)).astype(float)
    batched = scorer.score_batch(W)
    one_by_one = np.stack([scorer.score_batch(w[None, :])[0] for w in W])
    np.testing.assert_allclose(batched, one_by_one, rtol=1e-12, atol=1e-12)


def test_prepared_scorer_unbatched_path_matches(ga_inputs):
    '''With no memory for the outer products, the fallback must agree.'''
    rnaseq, ppi, _ = ga_inputs
    bi_ppi = bidirectional_ppi_for_cci(ppi, verbose=False)
    space = InteractionSpace(rnaseq_data=rnaseq, ppi_data=bi_ppi,
                             gene_cutoffs={'type': 'constant_value', 'parameter': 10},
                             communication_score='expression_thresholding',
                             cci_score='bray_curtis', cci_type='undirected', verbose=False)
    batched = ga.PreparedCCIScorer(space, max_memory_mb=1e9)
    unbatched = ga.PreparedCCIScorer(space, max_memory_mb=0)
    assert batched.batched and not unbatched.batched

    W = np.random.default_rng(3).integers(0, 2, size=(4, batched.n_ppi)).astype(float)
    np.testing.assert_allclose(batched.score_batch(W), unbatched.score_batch(W),
                               rtol=1e-12, atol=1e-12)


def test_prepared_scorer_symmetric_for_undirected(ga_inputs):
    rnaseq, ppi, _ = ga_inputs
    bi_ppi = bidirectional_ppi_for_cci(ppi, verbose=False)
    space = InteractionSpace(rnaseq_data=rnaseq, ppi_data=bi_ppi,
                             gene_cutoffs={'type': 'constant_value', 'parameter': 10},
                             communication_score='expression_thresholding',
                             cci_score='bray_curtis', cci_type='undirected', verbose=False)
    scorer = ga.PreparedCCIScorer(space)
    matrix = scorer.score(np.ones(scorer.n_ppi))
    assert list(matrix.index) == list(matrix.columns) == scorer.cell_names
    np.testing.assert_allclose(matrix.values, matrix.values.T, rtol=1e-12, atol=1e-12)


def test_prepared_scorer_rejects_unsupported_score(ga_inputs):
    rnaseq, ppi, _ = ga_inputs
    bi_ppi = bidirectional_ppi_for_cci(ppi, verbose=False)
    space = InteractionSpace(rnaseq_data=rnaseq, ppi_data=bi_ppi,
                             gene_cutoffs={'type': 'constant_value', 'parameter': 10},
                             communication_score='expression_thresholding',
                             cci_score='bray_curtis', cci_type='undirected', verbose=False)
    with pytest.raises(NotImplementedError):
        ga.PreparedCCIScorer(space, cci_score='cosine')


def test_prepared_scorer_count_rejects_non_binary_weights(ga_inputs):
    '''`count` counts non-zero products, so it is only linear in binary weights.'''
    rnaseq, ppi, _ = ga_inputs
    bi_ppi = bidirectional_ppi_for_cci(ppi, verbose=False)
    space = InteractionSpace(rnaseq_data=rnaseq, ppi_data=bi_ppi,
                             gene_cutoffs={'type': 'constant_value', 'parameter': 10},
                             communication_score='expression_thresholding',
                             cci_score='count', cci_type='undirected', verbose=False)
    scorer = ga.PreparedCCIScorer(space)
    with pytest.raises(ValueError):
        scorer.score_batch(np.full((1, scorer.n_ppi), 0.5))


# ---------------------------------------------------------------------------------
# _bidirectional_index
# ---------------------------------------------------------------------------------

def test_bidirectional_index_recovers_the_weights(toy_ppi):
    '''Expanding a per-pair vector must equal bidirectionalizing the table itself.'''
    pool = remove_ppi_bidirectionality(toy_ppi, interaction_columns=('A', 'B'),
                                       verbose=False).reset_index(drop=True)
    source = ga._bidirectional_index(pool, verbose=False)
    theta = np.arange(len(pool), dtype=float) + 1.0

    expected = bidirectional_ppi_for_cci(pool.assign(score=theta), verbose=False)
    np.testing.assert_array_equal(theta[source], expected['score'].values)


def test_bidirectional_index_length_is_independent_of_the_weights(toy_ppi):
    '''On a deduplicated pool the bidirectional table has a fixed number of rows.'''
    pool = remove_ppi_bidirectionality(toy_ppi, interaction_columns=('A', 'B'),
                                       verbose=False).reset_index(drop=True)
    source = ga._bidirectional_index(pool, verbose=False)
    n_self = int((pool['A'] == pool['B']).sum())
    assert len(source) == 2 * len(pool) - n_self

    rng = np.random.default_rng(0)
    for _ in range(5):
        theta = rng.integers(0, 2, size=len(pool)).astype(float)
        bi = bidirectional_ppi_for_cci(pool.assign(score=theta), verbose=False)
        assert len(bi) == len(source)


def test_bidirectional_index_rejects_reciprocal_pairs(toy_ppi):
    '''`toy_ppi` holds A-B and B-A, which makes the mapping weight-dependent.'''
    assert len(toy_ppi) > len(remove_ppi_bidirectionality(
        toy_ppi, interaction_columns=('A', 'B'), verbose=False))
    with pytest.raises(ValueError, match='depends on the weights'):
        ga._bidirectional_index(toy_ppi, verbose=False)


def test_optimize_lr_pairs_deduplicates_by_default(ga_inputs, setups):
    '''With deduplicate=True a pool holding both directions is accepted.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    reciprocal = pd.concat([ppi, ppi.rename(columns={'A': 'B', 'B': 'A'})[['A', 'B']]],
                           ignore_index=True)
    results = c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq, ppi_data=reciprocal,
                                             reference_distances=reference,
                                             cutoff_setup=cutoff_setup,
                                             analysis_setup=analysis_setup,
                                             population_size=8, generations=3, runs=1,
                                             random_state=5)
    assert 0.0 <= results['best_obj_fn'] <= 1.0


# ---------------------------------------------------------------------------------
# optimize_lr_pairs
# ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_optimize_lr_pairs_runs_and_selects(ga_inputs, setups):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    results = c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq, ppi_data=ppi,
                                             reference_distances=reference,
                                             cutoff_setup=cutoff_setup,
                                             analysis_setup=analysis_setup,
                                             population_size=20, generations=8, runs=2,
                                             random_state=888)
    assert set(['run1', 'run2']).issubset(results.keys())
    for key in ('run1', 'run2'):
        assert 0.0 <= results[key]['obj_fn'] <= 1.0
        assert len(results[key]['ppi_data']) == len(ppi)
        assert set(results[key]['ppi_data']).issubset({0, 1})
        assert 0.0 <= results[key]['drop_fraction'] <= 1.0
    assert results['best_obj_fn'] == max(results[k]['obj_fn'] for k in ('run1', 'run2'))
    assert len(results['best_ppi_data']) == results[results['best_run']]['n_selected']


@pytest.mark.slow
def test_optimize_lr_pairs_fast_equals_reference(ga_inputs, setups):
    '''The vectorized objective must not change which pairs get selected.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    kwargs = dict(rnaseq_data=rnaseq, ppi_data=ppi, reference_distances=reference,
                  cutoff_setup=cutoff_setup, analysis_setup=analysis_setup,
                  population_size=16, generations=6, runs=2, random_state=42)

    fast = c2c.analysis.optimize_lr_pairs(fast=True, **kwargs)
    slow = c2c.analysis.optimize_lr_pairs(fast=False, **kwargs)

    for key in ('run1', 'run2'):
        assert np.isclose(fast[key]['obj_fn'], slow[key]['obj_fn'], rtol=1e-9, atol=1e-9)
        assert fast[key]['ppi_data'] == slow[key]['ppi_data']


@pytest.mark.slow
def test_optimize_lr_pairs_is_reproducible(ga_inputs, setups):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    kwargs = dict(rnaseq_data=rnaseq, ppi_data=ppi, reference_distances=reference,
                  cutoff_setup=cutoff_setup, analysis_setup=analysis_setup,
                  population_size=16, generations=6, runs=1, random_state=7)
    first = c2c.analysis.optimize_lr_pairs(**kwargs)
    second = c2c.analysis.optimize_lr_pairs(**kwargs)
    assert first['run1']['ppi_data'] == second['run1']['ppi_data']
    assert np.isclose(first['run1']['obj_fn'], second['run1']['obj_fn'])


@pytest.mark.slow
def test_optimize_lr_pairs_shrinks_the_pair_set(ga_inputs, setups):
    '''Each run searches only among the pairs the previous one kept.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    results = c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq, ppi_data=ppi,
                                             reference_distances=reference,
                                             cutoff_setup=cutoff_setup,
                                             analysis_setup=analysis_setup,
                                             population_size=16, generations=6, runs=3,
                                             random_state=13)
    counts = [results['run{}'.format(i)]['n_selected'] for i in (1, 2, 3)]
    assert counts[0] >= counts[1] >= counts[2]


def test_optimize_lr_pairs_rejects_directed(ga_inputs, setups):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    analysis_setup = dict(analysis_setup, cci_type='directed')
    with pytest.raises(NotImplementedError):
        c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq, ppi_data=ppi,
                                       reference_distances=reference,
                                       cutoff_setup=cutoff_setup,
                                       analysis_setup=analysis_setup,
                                       population_size=8, generations=2, runs=1)


def test_optimize_lr_pairs_rejects_asymmetric_reference(ga_inputs, setups):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    asymmetric = reference.copy()
    asymmetric.iloc[0, 1] = asymmetric.iloc[1, 0] + 1.0
    with pytest.raises(ValueError):
        c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq, ppi_data=ppi,
                                       reference_distances=asymmetric,
                                       cutoff_setup=cutoff_setup,
                                       analysis_setup=analysis_setup,
                                       population_size=8, generations=2, runs=1)


# ---------------------------------------------------------------------------------
# Integrating independent executions
# ---------------------------------------------------------------------------------

def test_lr_selection_frequency():
    masks = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0]])
    np.testing.assert_allclose(ga.lr_selection_frequency(masks), [0.75, 0.25, 0.25])


def test_lr_cooccurrence_is_the_jaccard_of_selection_patterns():
    # pair 0 and 1 are always chosen together; pair 2 never with either
    masks = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    co = ga.lr_cooccurrence(masks, labels=['a', 'b', 'c'])
    assert np.isclose(co.loc['a', 'b'], 1.0)
    assert np.isclose(co.loc['a', 'c'], 0.0)
    assert np.isclose(co.loc['a', 'a'], 1.0)
    np.testing.assert_allclose(co.values, co.values.T)


def test_lr_cooccurrence_matches_the_reference_loop():
    '''Pinned against the double loop of the published notebook.'''
    rng = np.random.default_rng(0)
    masks = rng.integers(0, 2, size=(12, 20))
    masks[:, 3] = 0                                    # a pair no run ever selected
    labels = ['lr{}'.format(i) for i in range(20)]
    df = pd.DataFrame(masks, columns=labels).astype(int)

    expected = pd.DataFrame(np.zeros((len(labels), len(labels))),
                            columns=labels, index=labels)
    for i, lr in enumerate(labels):
        v1 = df[lr]
        for lr2 in labels[i:]:
            v2 = df[lr2]
            union = sum(v1.values | v2.values)
            val = 0.0 if union == 0 else sum(v1.values & v2.values) / union
            expected.at[lr, lr2] = val
            expected.at[lr2, lr] = val

    got = ga.lr_cooccurrence(masks, labels=labels)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12, atol=1e-12)


def test_lr_cooccurrence_zero_for_never_selected_pairs():
    masks = np.array([[1, 0], [1, 0]])
    co = ga.lr_cooccurrence(masks, labels=['chosen', 'never'])
    assert (co.loc['never'] == 0).all()


def test_consensus_from_cooccurrence_picks_the_co_selected_group():
    '''Two blocks: a tight one of 3 pairs and a loose one of 5.'''
    tight, loose = 3, 5
    n = tight + loose
    rng = np.random.default_rng(0)
    masks = np.zeros((20, n), dtype=int)
    masks[:, :tight] = 1                                     # always together
    masks[:, tight:] = rng.integers(0, 2, size=(20, loose))   # independently
    labels = ['t{}'.format(i) for i in range(tight)] + ['l{}'.format(i) for i in range(loose)]

    co = ga.lr_cooccurrence(masks, labels=labels)
    selected, clusters, scores = ga.consensus_from_cooccurrence(co, n_clusters=2)
    assert set(selected) == {'t0', 't1', 't2'}
    assert max(scores, key=lambda k: scores[k]) in clusters
    # the chosen cluster is the one whose members co-occur most
    chosen_score = max(scores.values())
    assert chosen_score == scores[[k for k, v in clusters.items() if set(v) == set(selected)][0]]


def test_consensus_from_cooccurrence_smallest_reproduces_the_notebook():
    masks = np.zeros((10, 8), dtype=int)
    masks[:, :2] = 1
    masks[:, 2:] = np.random.default_rng(1).integers(0, 2, size=(10, 6))
    labels = ['lr{}'.format(i) for i in range(8)]
    co = ga.lr_cooccurrence(masks, labels=labels)
    smallest, clusters, _ = ga.consensus_from_cooccurrence(co, select='smallest')
    assert len(smallest) == min(len(v) for v in clusters.values())


def test_consensus_from_cooccurrence_rejects_bad_select():
    co = ga.lr_cooccurrence(np.ones((3, 4), dtype=int))
    with pytest.raises(ValueError):
        ga.consensus_from_cooccurrence(co, select='not_an_option')


def test_consensus_from_cooccurrence_needs_enough_selected_pairs():
    masks = np.zeros((5, 6), dtype=int)
    masks[:, 0] = 1                       # only one pair ever selected
    with pytest.raises(ValueError, match='fewer than'):
        ga.consensus_from_cooccurrence(ga.lr_cooccurrence(masks), n_clusters=2)


def test_consensus_from_frequency():
    frequency = np.array([0.1, 0.2, 0.9, 1.0, 0.05])
    mask, threshold = ga.consensus_from_frequency(frequency, percentile=60)
    assert mask.sum() == 2 and mask[2] and mask[3]
    assert np.isclose(threshold, np.percentile(frequency, 60))


@pytest.mark.slow
def test_optimize_lr_pairs_multiple_executions(ga_inputs, setups):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    results = c2c.analysis.optimize_lr_pairs(
        rnaseq_data=rnaseq, ppi_data=ppi, reference_distances=reference,
        cutoff_setup=cutoff_setup, analysis_setup=analysis_setup,
        executions=4, population_size=16, generations=5, runs=1, random_state=11)

    assert len(results['executions']) == 4
    assert results['selection_masks'].shape == (4, len(ppi))
    assert set(np.unique(results['selection_masks'])).issubset({0, 1})
    assert len(results['selection_frequency']) == len(ppi)
    assert results['cooccurrence'].shape == (len(ppi), len(ppi))
    assert results['consensus_ppi_data'] is not None
    assert len(results['consensus_ppi_data']) <= len(ppi)
    # the reported best is the best of the executions
    assert np.isclose(results['best_obj_fn'],
                      max(e['best_obj_fn'] for e in results['executions'].values()))


@pytest.mark.slow
def test_optimize_lr_pairs_frequency_consensus(ga_inputs, setups):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    results = c2c.analysis.optimize_lr_pairs(
        rnaseq_data=rnaseq, ppi_data=ppi, reference_distances=reference,
        cutoff_setup=cutoff_setup, analysis_setup=analysis_setup,
        executions=4, consensus_method='frequency', frequency_percentile=75,
        population_size=16, generations=5, runs=1, random_state=11)
    assert 'consensus_threshold' in results
    frequency = results['selection_frequency']['frequency'].values
    assert len(results['consensus_ppi_data']) == int((frequency > results['consensus_threshold']).sum())


@pytest.mark.slow
def test_optimize_lr_pairs_executions_are_reproducible(ga_inputs, setups):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    kwargs = dict(rnaseq_data=rnaseq, ppi_data=ppi, reference_distances=reference,
                  cutoff_setup=cutoff_setup, analysis_setup=analysis_setup,
                  executions=3, population_size=16, generations=5, runs=1, random_state=5)
    first = c2c.analysis.optimize_lr_pairs(**kwargs)
    second = c2c.analysis.optimize_lr_pairs(**kwargs)
    np.testing.assert_array_equal(first['selection_masks'], second['selection_masks'])


def test_optimize_lr_pairs_rejects_bad_consensus_method(ga_inputs, setups):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    with pytest.raises(ValueError):
        c2c.analysis.optimize_lr_pairs(
            rnaseq_data=rnaseq, ppi_data=ppi, reference_distances=reference,
            cutoff_setup=cutoff_setup, analysis_setup=analysis_setup,
            executions=2, consensus_method='nope', population_size=8,
            generations=2, runs=1, random_state=1)


# ---------------------------------------------------------------------------------
# Pluggable objectives
# ---------------------------------------------------------------------------------

@pytest.fixture
def objective_factory(ga_inputs, setups):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    return c2c.analysis.CorrelationObjective(rnaseq_data=rnaseq, reference_distances=reference,
                                             cutoff_setup=cutoff_setup,
                                             analysis_setup=analysis_setup)


def _second_dataset(seed, n_cells=10):
    from sklearn.metrics.pairwise import euclidean_distances
    genes = ['G{}'.format(i) for i in range(150)]
    rnaseq = c2c.datasets.generate_random_rnaseq(size=n_cells, row_names=genes,
                                                 random_state=seed, verbose=False)
    coords = np.random.default_rng(seed).random((n_cells, 2)) * 100
    reference = pd.DataFrame(euclidean_distances(coords, coords),
                             index=rnaseq.columns, columns=rnaseq.columns)
    return rnaseq, reference


def test_correlation_fitness_default_is_absolute():
    assert np.isclose(c2c.analysis.correlation_fitness([1, 2, 3], [3, 2, 1]), 1.0)


def test_correlation_fitness_signed_keeps_the_sign():
    assert np.isclose(c2c.analysis.correlation_fitness([1, 2, 3], [3, 2, 1], signed=True), -1.0)


def test_correlation_fitness_nan_becomes_zero():
    assert c2c.analysis.correlation_fitness([1, 1, 1], [1, 2, 3]) == 0.0


@pytest.mark.slow
def test_explicit_objective_matches_the_default(ga_inputs, setups, objective_factory):
    '''Passing the default objective explicitly must change nothing.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    kwargs = dict(ppi_data=ppi, population_size=16, generations=5, runs=1, random_state=7)

    implicit = c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq, reference_distances=reference,
                                              cutoff_setup=cutoff_setup,
                                              analysis_setup=analysis_setup, **kwargs)
    explicit = c2c.analysis.optimize_lr_pairs(objective=objective_factory, **kwargs)
    assert implicit['run1']['ppi_data'] == explicit['run1']['ppi_data']
    assert np.isclose(implicit['run1']['obj_fn'], explicit['run1']['obj_fn'])


@pytest.mark.slow
def test_combined_objective_over_duplicates_equals_one(ga_inputs, setups, objective_factory):
    '''Combining a dataset with itself must reduce to the single-dataset case.'''
    _, ppi, _ = ga_inputs
    kwargs = dict(ppi_data=ppi, population_size=16, generations=5, runs=1, random_state=7)
    single = c2c.analysis.optimize_lr_pairs(objective=objective_factory, **kwargs)
    doubled = c2c.analysis.optimize_lr_pairs(
        objective=c2c.analysis.CombinedObjective([objective_factory, objective_factory]),
        **kwargs)
    assert single['run1']['ppi_data'] == doubled['run1']['ppi_data']


@pytest.mark.slow
def test_combined_objective_over_several_datasets(ga_inputs, setups):
    _, ppi, _ = ga_inputs
    analysis_setup, cutoff_setup = setups
    factories = [c2c.analysis.CorrelationObjective(rnaseq_data=r, reference_distances=d,
                                                   cutoff_setup=cutoff_setup,
                                                   analysis_setup=analysis_setup)
                 for r, d in (_second_dataset(0), _second_dataset(1), _second_dataset(2))]
    results = c2c.analysis.optimize_lr_pairs(
        ppi_data=ppi, objective=c2c.analysis.CombinedObjective(factories),
        population_size=16, generations=5, runs=1, random_state=7)
    assert 0.0 <= results['best_obj_fn'] <= 1.0
    assert len(results['run1']['ppi_data']) == len(ppi)


def test_combined_objective_components_and_penalty(ga_inputs, setups):
    '''The penalty must subtract exactly sd_penalty * population SD of the components.'''
    _, ppi, _ = ga_inputs
    analysis_setup, cutoff_setup = setups
    pool = remove_ppi_bidirectionality(ppi, interaction_columns=('A', 'B'), verbose=False)
    pool = pool.drop_duplicates(subset=['A', 'B']).reset_index(drop=True)
    factories = [c2c.analysis.CorrelationObjective(rnaseq_data=r, reference_distances=d,
                                                   cutoff_setup=cutoff_setup,
                                                   analysis_setup=analysis_setup)
                 for r, d in (_second_dataset(0), _second_dataset(1), _second_dataset(2))]

    plain = c2c.analysis.CombinedObjective(factories)(pool)
    penalised = c2c.analysis.CombinedObjective(factories, sd_penalty=0.5)(pool)

    masks = np.random.default_rng(0).integers(0, 2, size=(3, len(pool))).astype(float)
    components = plain.evaluate_components(masks)
    assert components.shape == (3, 3)
    np.testing.assert_allclose(plain(masks), components.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(penalised(masks),
                               components.mean(axis=0) - 0.5 * components.std(axis=0, ddof=0),
                               rtol=1e-12)


def test_combined_objective_single_dataset_has_zero_penalty(ga_inputs, setups, objective_factory):
    '''With one dataset the SD is 0, so the penalty cannot change anything.'''
    _, ppi, _ = ga_inputs
    pool = remove_ppi_bidirectionality(ppi, interaction_columns=('A', 'B'), verbose=False)
    pool = pool.drop_duplicates(subset=['A', 'B']).reset_index(drop=True)
    masks = np.random.default_rng(0).integers(0, 2, size=(3, len(pool))).astype(float)
    plain = c2c.analysis.CombinedObjective([objective_factory])(pool)
    heavy = c2c.analysis.CombinedObjective([objective_factory], sd_penalty=10.0)(pool)
    np.testing.assert_allclose(plain(masks), heavy(masks), rtol=1e-12)


@pytest.mark.parametrize('combine,expected', [('mean', 2.0), ('median', 2.0),
                                              ('min', 1.0), ('max', 3.0)])
def test_combined_objective_combiners(combine, expected):
    class Fixed:
        def __init__(self, value): self.value = value
        def __call__(self, pool): return lambda masks: np.full(len(np.atleast_2d(masks)), self.value)

    objective = c2c.analysis.CombinedObjective([Fixed(1.0), Fixed(2.0), Fixed(3.0)],
                                               combine=combine)(pd.DataFrame({'A': ['a']}))
    assert np.isclose(objective(np.zeros((1, 1)))[0], expected)


def test_combined_objective_accepts_a_callable_combiner():
    class Fixed:
        def __init__(self, value): self.value = value
        def __call__(self, pool): return lambda masks: np.full(len(np.atleast_2d(masks)), self.value)

    objective = c2c.analysis.CombinedObjective(
        [Fixed(1.0), Fixed(3.0)],
        combine=lambda values, weights=None: values.sum(axis=0))(pd.DataFrame({'A': ['a']}))
    assert np.isclose(objective(np.zeros((1, 1)))[0], 4.0)


def test_combined_objective_validates_its_arguments(objective_factory):
    with pytest.raises(ValueError):
        c2c.analysis.CombinedObjective([])
    with pytest.raises(ValueError):
        c2c.analysis.CombinedObjective([objective_factory], sd_penalty=-1.0)
    with pytest.raises(ValueError):
        c2c.analysis.CombinedObjective([objective_factory], combine='not_a_combiner')
    with pytest.raises(ValueError):
        c2c.analysis.CombinedObjective([objective_factory], weights=[1.0, 2.0])


def test_optimize_lr_pairs_rejects_objective_and_data_together(ga_inputs, setups,
                                                               objective_factory):
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    with pytest.raises(ValueError, match='not both'):
        c2c.analysis.optimize_lr_pairs(rnaseq_data=rnaseq, ppi_data=ppi,
                                       reference_distances=reference,
                                       cutoff_setup=cutoff_setup,
                                       analysis_setup=analysis_setup,
                                       objective=objective_factory,
                                       population_size=8, generations=2, runs=1)


def test_optimize_lr_pairs_requires_objective_or_data(ga_inputs):
    _, ppi, _ = ga_inputs
    with pytest.raises(ValueError, match='required'):
        c2c.analysis.optimize_lr_pairs(ppi_data=ppi, population_size=8, generations=2, runs=1)


@pytest.mark.slow
def test_custom_objective_can_penalise_size(ga_inputs, setups, objective_factory):
    '''A custom objective wrapping the default -- the documented extension pattern.'''
    _, ppi, _ = ga_inputs

    class SparsityPenalised:
        def __init__(self, inner, penalty): self.inner, self.penalty = inner, penalty
        def __call__(self, pool):
            bound = self.inner(pool)
            def objective(masks):
                masks = np.atleast_2d(np.asarray(masks, dtype=float))
                return bound(masks) - self.penalty * masks.sum(axis=1) / masks.shape[1]
            return objective

    plain = c2c.analysis.optimize_lr_pairs(ppi_data=ppi, objective=objective_factory,
                                           population_size=16, generations=6, runs=1,
                                           random_state=3)
    sparse = c2c.analysis.optimize_lr_pairs(
        ppi_data=ppi, objective=SparsityPenalised(objective_factory, penalty=1.0),
        population_size=16, generations=6, runs=1, random_state=3)
    assert sparse['run1']['n_selected'] < plain['run1']['n_selected']


@pytest.mark.slow
def test_selection_masks_align_with_the_pool(ga_inputs, setups):
    '''`selection_frequency` is sorted, so only `pool` is safe to pair with the masks.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    results = c2c.analysis.optimize_lr_pairs(
        rnaseq_data=rnaseq, ppi_data=ppi, reference_distances=reference,
        cutoff_setup=cutoff_setup, analysis_setup=analysis_setup,
        executions=4, population_size=12, generations=4, runs=1, random_state=1)

    masks = np.asarray(results['selection_masks'])
    assert masks.shape[1] == len(results['pool'])
    # column j of the masks is row j of the pool
    np.testing.assert_allclose(masks.mean(axis=0),
                               results['selection_frequency'].sort_index()['frequency'].values)
    # and the frequency frame keeps the pool index, so it can be realigned
    realigned = results['selection_frequency'].sort_index()
    pd.testing.assert_frame_equal(realigned[['A', 'B']].reset_index(drop=True),
                                  results['pool'][['A', 'B']].reset_index(drop=True))
