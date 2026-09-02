# -*- coding: utf-8 -*-

'''Tests for cell2cell.analysis.genetic_algorithm'''

import json
import pathlib
import urllib.request

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.analysis import genetic_algorithm as ga
from cell2cell.core.interaction_space import InteractionSpace
from cell2cell.preprocessing.ppi import (bidirectional_ppi_for_cci,
                                         bidirectional_ppi_with_index,
                                         remove_ppi_bidirectionality)

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


def test_prepared_scorer_unbatched_count_counts_instead_of_summing(ga_inputs):
    '''`count` must not depend on whether the outer products fitted in memory.

    With continuous expression the products are not 0/1, so adding them up instead of
    counting the non-zero ones gives a different, and much larger, answer.
    '''
    rnaseq, ppi, _ = ga_inputs
    bi_ppi = bidirectional_ppi_for_cci(ppi, verbose=False)
    space = InteractionSpace(rnaseq_data=rnaseq, ppi_data=bi_ppi,
                             gene_cutoffs={'type': 'constant_value', 'parameter': 10},
                             communication_score='expression_product',
                             cci_score='count', cci_type='undirected', verbose=False)
    batched = ga.PreparedCCIScorer(space, cci_score='count', max_memory_mb=1e9)
    unbatched = ga.PreparedCCIScorer(space, cci_score='count', max_memory_mb=0)
    assert batched.batched and not unbatched.batched

    source = ga._bidirectional_index(ppi, verbose=False)
    weights = np.random.default_rng(4).integers(0, 2, size=len(ppi)).astype(float)[source]

    space.ppi_data['score'] = weights
    space.compute_pairwise_cci_scores(use_ppi_score=True, verbose=False)
    expected = space.interaction_elements['cci_matrix'].values.astype(float)

    np.testing.assert_allclose(batched.score_batch(weights[None, :])[0], expected,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(unbatched.score_batch(weights[None, :])[0], expected,
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


@pytest.mark.parametrize('max_memory_mb', [512, 0])
def test_prepared_scorer_count_rejects_non_binary_weights(ga_inputs, max_memory_mb):
    '''`count` counts non-zero products, so it is only linear in binary weights.

    Parametrized over the memory budget because the rejection has to be the same on
    the batched and the per-vector path.
    '''
    rnaseq, ppi, _ = ga_inputs
    bi_ppi = bidirectional_ppi_for_cci(ppi, verbose=False)
    space = InteractionSpace(rnaseq_data=rnaseq, ppi_data=bi_ppi,
                             gene_cutoffs={'type': 'constant_value', 'parameter': 10},
                             communication_score='expression_thresholding',
                             cci_score='count', cci_type='undirected', verbose=False)
    scorer = ga.PreparedCCIScorer(space, max_memory_mb=max_memory_mb)
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


def test_bidirectional_index_handles_reciprocal_pairs(toy_ppi):
    '''`toy_ppi` holds A-B and B-A. Provenance is recorded while the table is built,
    so it is well defined regardless -- it never has to be inferred from the scores.'''
    assert len(toy_ppi) > len(remove_ppi_bidirectionality(
        toy_ppi, interaction_columns=('A', 'B'), verbose=False))
    table, origin = bidirectional_ppi_with_index(toy_ppi, verbose=False)
    assert len(origin) == len(table)
    assert set(origin).issubset(set(range(len(toy_ppi))))


def test_bidirectional_table_keeps_repeated_pairs_with_different_metadata():
    '''Two rows for the same partners are two interactions, not one.

    They differ only outside the interacting columns -- a score, an annotation -- so
    deduplicating on those columns alone would drop one of them and leave its
    position absent from `origin`, making it unselectable.
    '''
    ppi = pd.DataFrame({'A': ['a', 'a', 'c'], 'B': ['b', 'b', 'd'],
                        'score': [0.5, 1.0, 1.0],
                        'source': ['curated', 'predicted', 'curated']})
    table, origin = bidirectional_ppi_with_index(ppi, verbose=False)

    expected = bidirectional_ppi_for_cci(ppi, verbose=False)
    pd.testing.assert_frame_equal(table, expected)
    assert set(origin) == set(range(len(ppi)))
    np.testing.assert_array_equal(table['score'].values, ppi['score'].values[origin])


def test_bidirectional_table_matches_the_original_builder(toy_ppi, toy_ppi_complex):
    '''The constructed table must equal what `bidirectional_ppi_for_cci` returns.'''
    for ppi in (toy_ppi, toy_ppi_complex):
        ppi = ppi.copy()
        if 'score' not in ppi.columns:
            ppi = ppi.assign(score=1.0)
        expected = bidirectional_ppi_for_cci(ppi, verbose=False)
        got, _ = bidirectional_ppi_with_index(ppi, verbose=False)
        pd.testing.assert_frame_equal(got[['A', 'B']].reset_index(drop=True),
                                      expected[['A', 'B']].reset_index(drop=True))


def test_bidirectional_index_expands_a_per_pair_vector(toy_ppi):
    '''Each bidirectional row takes the value of the interaction it came from.'''
    table, origin = bidirectional_ppi_with_index(toy_ppi, verbose=False)
    theta = np.arange(len(toy_ppi), dtype=float) + 1.0
    expanded = theta[origin]
    for row, value in zip(table[['A', 'B']].values, expanded):
        source = toy_ppi.iloc[int(np.where(theta == value)[0][0])]
        assert set(row) == set(source[['A', 'B']].values)


def test_self_interactions_are_not_counted_twice(toy_ppi):
    '''Doubling a self-interaction would weight that protein pairing double.'''
    table, origin = bidirectional_ppi_with_index(toy_ppi, verbose=False)
    self_rows = toy_ppi[toy_ppi['A'] == toy_ppi['B']]
    assert len(self_rows) > 0
    for position in self_rows.index:
        assert (origin == position).sum() == 1     # once, not twice


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
# Automatic stopping
# ---------------------------------------------------------------------------------

class _ScriptedObjective:
    '''Objective factory whose fitness is a fixed value per run, ignoring the mask.

    Lets the stopping rule be exercised on an exact sequence of objective values,
    including the negative and zero ones a signed correlation or a custom objective
    can produce.
    '''

    def __init__(self, values):
        self.values = list(values)
        self.runs = 0

    def __call__(self, pool):
        value = float(self.values[min(self.runs, len(self.values) - 1)])
        self.runs += 1

        def objective(masks):
            masks = np.atleast_2d(np.asarray(masks, dtype=float))
            return np.full(len(masks), value)

        return objective


@pytest.mark.parametrize('best,previous,expected', [
    (0.55, 0.50, 0.1),          # the positive case, unchanged
    (-0.40, -0.50, 0.2),        # an improvement, even though both are negative
    (-0.60, -0.50, -0.2),       # a worsening
    (0.50, 0.0, 0.5 / 1e-12),   # no division by zero
    (0.0, 0.0, 0.0),
])
def test_relative_improvement(best, previous, expected):
    assert np.isclose(ga.search._relative_improvement(best, previous), expected)


def test_runs_continue_while_a_negative_objective_improves(ga_inputs):
    '''-0.50 to -0.40 is a 20% improvement, not a stall.'''
    _, ppi, _ = ga_inputs
    scripted = _ScriptedObjective([-0.50, -0.40, -0.399])
    results = c2c.analysis.optimize_lr_pairs(ppi_data=ppi, objective=scripted,
                                             population_size=8, generations=2,
                                             runs=None, inc_percentage=0.025,
                                             random_state=11)
    assert 'run3' in results
    assert np.isclose(results['run1']['obj_fn'], -0.50)
    assert np.isclose(results['run2']['obj_fn'], -0.40)


def test_an_objective_of_zero_stops_rather_than_dividing_by_it(ga_inputs):
    '''A run scoring exactly zero used to be divided by.

    The division gave NaN, which compares false against the threshold, so the search
    carried on running instead of recognizing that nothing had improved.
    '''
    _, ppi, _ = ga_inputs
    results = c2c.analysis.optimize_lr_pairs(ppi_data=ppi,
                                             objective=_ScriptedObjective([0.0]),
                                             population_size=8, generations=2,
                                             runs=None, inc_percentage=0.025,
                                             random_state=11)
    assert 'run2' in results and 'run3' not in results
    assert results['best_obj_fn'] == 0.0


# ---------------------------------------------------------------------------------
# An interaction weight is not a selection
# ---------------------------------------------------------------------------------

class _RecordingObjective:
    '''Objective factory that remembers every pool it is handed.

    Its fitness depends only on the mask, which is all the search requires, so the
    pools can be inspected without running a real objective.
    '''

    def __init__(self):
        self.pools = []

    def __call__(self, pool):
        self.pools.append(pool.copy())
        n_pool = len(pool)

        def objective(masks):
            masks = np.atleast_2d(np.asarray(masks, dtype=float))
            return masks.sum(axis=1) / n_pool

        return objective


def _weighted(ppi):
    '''The same pairs, carrying interaction weights other than one.'''
    return ppi.assign(score=np.resize([0.5, 0.8, 2.0, 1.0], len(ppi)).astype(float))


def test_optimize_lr_pairs_searches_every_weighted_pair(ga_inputs):
    '''A weight is not a selection, so no pair may be dropped before the search.'''
    _, ppi, _ = ga_inputs
    weighted = _weighted(ppi)
    assert (weighted['score'] != 1.0).sum() > 0

    recorder = _RecordingObjective()
    results = c2c.analysis.optimize_lr_pairs(ppi_data=weighted, objective=recorder,
                                             population_size=8, generations=2, runs=1,
                                             random_state=3)
    assert len(recorder.pools[0]) == len(results['pool'])
    assert len(results['run1']['ppi_data']) == len(results['pool'])


def test_optimize_lr_pairs_preserves_the_input_weights(ga_inputs):
    '''The search must not write its solution over the interaction weights.'''
    _, ppi, _ = ga_inputs
    weighted = _weighted(ppi)
    expected = {(a, b): s for a, b, s in weighted[['A', 'B', 'score']].values}

    results = c2c.analysis.optimize_lr_pairs(ppi_data=weighted,
                                             objective=_RecordingObjective(),
                                             population_size=8, generations=2, runs=2,
                                             random_state=3)
    for frame in (results['pool'], results['best_ppi_data']):
        for a, b, score in frame[['A', 'B', 'score']].values:
            assert score == expected[(a, b)]


@pytest.mark.parametrize('bad', [np.nan, -1.0])
def test_optimize_lr_pairs_rejects_unusable_weights(ga_inputs, bad):
    _, ppi, _ = ga_inputs
    weighted = _weighted(ppi)
    weighted.loc[weighted.index[0], 'score'] = bad
    with pytest.raises(ValueError):
        c2c.analysis.optimize_lr_pairs(ppi_data=weighted, objective=_RecordingObjective(),
                                       population_size=8, generations=2, runs=1)


def test_zero_weight_equals_not_being_selected(ga_inputs, setups):
    '''The weight scales what an interaction contributes; zero contributes nothing.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    factory = c2c.analysis.CorrelationObjective(rnaseq_data=rnaseq,
                                                reference_distances=reference,
                                                cutoff_setup=cutoff_setup,
                                                analysis_setup=analysis_setup)
    scores = np.ones(len(ppi))
    scores[3] = 0.0
    zero_weighted = factory(ppi.assign(score=scores))
    unit_weighted = factory(ppi.assign(score=1.0))

    everything = np.ones((1, len(ppi)))
    without_that_pair = everything.copy()
    without_that_pair[0, 3] = 0.0

    np.testing.assert_allclose(zero_weighted(everything),
                               unit_weighted(without_that_pair),
                               rtol=1e-12, atol=1e-12)


def test_unit_weights_are_the_same_as_no_weight_column(ga_inputs, setups):
    '''Weights of one must leave existing behaviour untouched.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    factory = c2c.analysis.CorrelationObjective(rnaseq_data=rnaseq,
                                                reference_distances=reference,
                                                cutoff_setup=cutoff_setup,
                                                analysis_setup=analysis_setup)
    masks = np.random.default_rng(6).integers(0, 2, size=(3, len(ppi))).astype(float)
    with_ones = factory(ppi.assign(score=1.0))(masks)
    without = factory(ppi.drop(columns='score', errors='ignore'))(masks)
    np.testing.assert_allclose(with_ones, without, rtol=1e-12, atol=1e-12)


def test_weighted_pool_agrees_between_the_fast_and_reference_paths(ga_inputs, setups):
    '''`validate_fast` compares both paths, so this fails if only one applies weights.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    factory = c2c.analysis.CorrelationObjective(rnaseq_data=rnaseq,
                                                reference_distances=reference,
                                                cutoff_setup=cutoff_setup,
                                                analysis_setup=analysis_setup,
                                                validate_fast=True)
    bound = factory(_weighted(ppi))          # raises if the two disagree
    masks = np.random.default_rng(7).integers(0, 2, size=(2, len(ppi))).astype(float)

    # Compared on the distances rather than the fitness: the fitness is a rank
    # correlation, and near-ties make ranks flip on differences of a few ULP
    fast = bound._distance_vectors(masks)
    reference_vectors = np.array([bound.reference_distance_vector(m) for m in masks])
    np.testing.assert_allclose(fast, reference_vectors, rtol=1e-9, atol=1e-9)


def test_repeated_pair_with_its_own_weight_is_selectable(ga_inputs, setups):
    '''With deduplicate=False a repeated pair stays, and each copy needs its own gene.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    pool = pd.concat([ppi.assign(score=1.0), ppi.iloc[[0]].assign(score=0.5)],
                     ignore_index=True)
    factory = c2c.analysis.CorrelationObjective(rnaseq_data=rnaseq,
                                                reference_distances=reference,
                                                cutoff_setup=cutoff_setup,
                                                analysis_setup=analysis_setup)
    bound = factory(pool)                     # both copies must be representable

    masks = np.ones((2, len(pool)))
    masks[1, -1] = 0.0                        # drop the repeated copy only
    vectors = bound._distance_vectors(masks)
    assert not np.allclose(vectors[0], vectors[1])


def test_deduplicated_pool_keeps_the_highest_weight_of_a_repeated_pair(ga_inputs):
    '''A pair listed twice is one interaction, and by default keeps its largest weight.'''
    _, ppi, _ = ga_inputs
    pool = pd.concat([ppi.assign(score=0.4), ppi.iloc[[0]].assign(score=0.9)],
                     ignore_index=True)

    recorder = _RecordingObjective()
    results = c2c.analysis.optimize_lr_pairs(ppi_data=pool, objective=recorder,
                                             population_size=8, generations=2, runs=1,
                                             random_state=3)
    searched = recorder.pools[0]
    assert len(searched) == len(ppi)                      # the repeat is collapsed
    repeated = (searched['A'] == ppi.iloc[0]['A']) & (searched['B'] == ppi.iloc[0]['B'])
    assert searched.loc[repeated, 'score'].tolist() == [0.9]
    assert len(results['pool']) == len(ppi)


def test_deduplicated_pool_can_keep_the_lowest_weight(ga_inputs):
    _, ppi, _ = ga_inputs
    pool = pd.concat([ppi.assign(score=0.4), ppi.iloc[[0]].assign(score=0.9)],
                     ignore_index=True)

    recorder = _RecordingObjective()
    c2c.analysis.optimize_lr_pairs(ppi_data=pool, objective=recorder, duplicates='lowest',
                                   population_size=8, generations=2, runs=1, random_state=3)
    searched = recorder.pools[0]
    repeated = (searched['A'] == ppi.iloc[0]['A']) & (searched['B'] == ppi.iloc[0]['B'])
    assert searched.loc[repeated, 'score'].tolist() == [0.4]


def test_returned_mask_selects_the_repeated_row_the_search_chose(ga_inputs):
    '''The mask reports positions, so one copy of a repeated pair can be dropped.

    Rebuilding it from the (A, B) names marks both copies whenever either is chosen,
    which contradicts the number of pairs the search reports selecting.
    '''
    _, ppi, _ = ga_inputs
    pool = pd.concat([ppi.assign(score=1.0), ppi.iloc[[0]].assign(score=0.5)],
                     ignore_index=True)

    class _PicksTheFirstCopy:
        '''Selects every pair except the repeated copy at the end of the pool.'''
        def __call__(self, candidate_pool):
            wanted = np.ones(len(candidate_pool))
            wanted[-1] = 0.0
            return lambda masks: -np.abs(np.atleast_2d(masks) - wanted).sum(axis=1)

    results = c2c.analysis.optimize_lr_pairs(ppi_data=pool, objective=_PicksTheFirstCopy(),
                                             deduplicate=False, population_size=30,
                                             generations=40, runs=1, random_state=4)
    mask = results['run1']['ppi_data']
    assert sum(mask) == results['run1']['n_selected']
    assert len(results['best_ppi_data']) == results['run1']['n_selected']


def test_cooccurrence_labels_stay_unique_for_a_repeated_pair(ga_inputs):
    '''Duplicate names would make the co-occurrence matrix ambiguous to index.'''
    _, ppi, _ = ga_inputs
    pool = pd.concat([ppi.assign(score=1.0), ppi.iloc[[0]].assign(score=0.5)],
                     ignore_index=True)
    results = c2c.analysis.optimize_lr_pairs(ppi_data=pool, objective=_RecordingObjective(),
                                             deduplicate=False, executions=3,
                                             population_size=8, generations=2, runs=1,
                                             random_state=4)
    labels = list(results['cooccurrence'].index)
    assert len(set(labels)) == len(labels) == len(pool)


def test_a_pool_holding_both_directions_is_refused(ga_inputs, setups):
    '''Doubling the table makes the two candidates share rows, so neither is separable.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    reversed_first = ppi.iloc[[0]].rename(columns={'A': 'B', 'B': 'A'})[['A', 'B']]
    pool = pd.concat([ppi, reversed_first], ignore_index=True)
    factory = c2c.analysis.CorrelationObjective(rnaseq_data=rnaseq,
                                                reference_distances=reference,
                                                cutoff_setup=cutoff_setup,
                                                analysis_setup=analysis_setup)
    with pytest.raises(ValueError, match='both directions'):
        factory(pool)


@pytest.mark.parametrize('bad', [np.inf, -np.inf])
def test_optimize_lr_pairs_rejects_infinite_weights(ga_inputs, bad):
    '''An infinite weight reaches the CCI scores as inf, or as NaN through inf/inf.'''
    _, ppi, _ = ga_inputs
    weighted = _weighted(ppi)
    weighted.loc[weighted.index[0], 'score'] = bad
    with pytest.raises(ValueError):
        c2c.analysis.optimize_lr_pairs(ppi_data=weighted, objective=_RecordingObjective(),
                                       population_size=8, generations=2, runs=1)


def test_reference_may_hold_missing_values_outside_the_cells_used(ga_inputs, setups):
    '''`included_cells` is the way out the error message offers, so it has to work.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    partial = reference.copy()
    excluded = list(reference.columns[-2:])
    partial.loc[excluded[0], excluded[1]] = np.nan
    partial.loc[excluded[1], excluded[0]] = np.nan

    kept = [c for c in reference.columns if c not in excluded]
    factory = c2c.analysis.CorrelationObjective(rnaseq_data=rnaseq,
                                                reference_distances=partial,
                                                cutoff_setup=cutoff_setup,
                                                analysis_setup=analysis_setup,
                                                included_cells=kept)
    assert np.isfinite(factory.reference_vector).all()

    with pytest.raises(ValueError, match='missing values off the diagonal'):
        c2c.analysis.CorrelationObjective(rnaseq_data=rnaseq, reference_distances=partial,
                                          cutoff_setup=cutoff_setup,
                                          analysis_setup=analysis_setup)


def test_pool_rows_that_repeat_exactly_are_refused(ga_inputs, setups):
    '''An exact duplicate cannot be selected on its own, so it is an error, not a no-op.'''
    rnaseq, ppi, reference = ga_inputs
    analysis_setup, cutoff_setup = setups
    pool = pd.concat([ppi, ppi.iloc[[0]]], ignore_index=True)
    factory = c2c.analysis.CorrelationObjective(rnaseq_data=rnaseq,
                                                reference_distances=reference,
                                                cutoff_setup=cutoff_setup,
                                                analysis_setup=analysis_setup)
    with pytest.raises(ValueError, match='repeat an earlier row'):
        factory(pool)


def test_weighted_pool_works_with_the_count_score(ga_inputs):
    '''`count` only asks whether an interaction is active, so weights must not trip it.'''
    rnaseq, ppi, reference = ga_inputs
    factory = c2c.analysis.CorrelationObjective(
        rnaseq_data=rnaseq, reference_distances=reference,
        cutoff_setup={'type': 'constant_value', 'parameter': 10},
        analysis_setup={'communication_score': 'expression_thresholding',
                        'cci_score': 'count', 'cci_type': 'undirected'})
    bound = factory(_weighted(ppi))
    values = bound(np.ones((1, len(ppi))))
    assert np.isfinite(values).all()


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
    # a single-execution result carries the pool too, so the two are consistent
    single = c2c.analysis.optimize_lr_pairs(
        rnaseq_data=rnaseq, ppi_data=ppi, reference_distances=reference,
        cutoff_setup=cutoff_setup, analysis_setup=analysis_setup,
        population_size=12, generations=4, runs=1, random_state=1)
    assert len(single['run1']['ppi_data']) == len(single['pool'])
    # column j of the masks is row j of the pool
    np.testing.assert_allclose(masks.mean(axis=0),
                               results['selection_frequency'].sort_index()['frequency'].values)
    # and the frequency frame keeps the pool index, so it can be realigned
    realigned = results['selection_frequency'].sort_index()
    pd.testing.assert_frame_equal(realigned[['A', 'B']].reset_index(drop=True),
                                  results['pool'][['A', 'B']].reset_index(drop=True))


# ---------------------------------------------------------------------------------
# Against the published C. elegans analysis
#
# The 100 genetic-algorithm executions of Armingol et al. (2022) are stored in
# LewisLabUCSD/Celegans-cell2cell, in the same shape this module returns. They pin the
# port against the analysis it reproduces: the same selection has to score the same
# fitness, and the same 100 selections have to give the same consensus list.
# `tests/data/make_celegans_ga_fixture.py` regenerates the fixture.
# ---------------------------------------------------------------------------------

CELEGANS_BASE = ('https://raw.githubusercontent.com/LewisLabUCSD/'
                 'Celegans-cell2cell/master/data/')
CELEGANS_COLUMNS = ('Ligand_symbol', 'Receptor_symbol')


@pytest.fixture(scope='module')
def published_executions():
    path = pathlib.Path(__file__).parent / 'data' / 'celegans_ga_consensus.npz'
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize('select', ['smallest', 'cooccurrence'])
def test_consensus_reproduces_the_published_lr_list(published_executions, select):
    '''The 100 published selections must give back the 37 pairs the paper reports.'''
    fixture = published_executions
    masks = fixture['masks'].astype(int)
    pairs = list(zip(fixture['ligands'], fixture['receptors']))
    labels = ['{}^{}^{}'.format(a, b, position) for position, (a, b) in enumerate(pairs)]

    frequency = c2c.analysis.lr_selection_frequency(masks)
    cooccurrence = c2c.analysis.lr_cooccurrence(masks, labels=labels)
    selected, _, _ = c2c.analysis.consensus_from_cooccurrence(
        cooccurrence, n_clusters=2, select=select, frequency=frequency, min_frequency=0.0)

    chosen = set(selected)
    got = {frozenset(pair) for label, pair in zip(labels, pairs) if label in chosen}
    expected = {frozenset(pair) for pair in zip(fixture['paper_ligands'],
                                                fixture['paper_receptors'])}
    assert got == expected


def test_published_masks_match_the_pool_they_are_indexed_against(published_executions):
    '''A positional mask only means anything against the ordering it was built on.'''
    fixture = published_executions
    assert fixture['masks'].shape == (100, len(fixture['ligands']))
    assert set(np.unique(fixture['masks'])) <= {0, 1}
    assert fixture['masks'].sum(axis=1).min() > 0


@pytest.mark.network
def test_the_two_paths_agree_on_the_published_data():
    '''The vectorized scorer must equal the reference path on the paper's own data.

    The stored `obj_fn` values are deliberately *not* the target here, and the reason
    is worth recording. Transcribing the original `optimal_ppi_score` from
    `code/genetic_algorithm.py` of the reference repository and running it on these
    same masks gives, over 20 runs of 5 executions, **exactly** what this objective
    gives -- to 1e-12, through both the reference and the vectorized path, with the
    bidirectional table the same 489 rows in both. The same 20 runs disagree with the
    `obj_fn` stored beside the mask 16 times. So the two implementations compute the
    same function of a selection, and it is the stored files that pair a mask with an
    objective belonging to a different selection. Which step of the original produced
    that pairing is not established here -- the objective and the mask are written two
    lines apart from the same `ga.bestIndividual()`, through a column that every
    candidate evaluation reassigns and a merge whose behaviour has changed across
    pandas versions -- and it does not need to be, since neither implementation
    disagrees about what a given selection scores.

    What the published data can pin here is that the two code paths agree on it, an
    equivalence otherwise only checked on synthetic fixtures. The published *result*,
    the 37 selected pairs, is pinned by
    `test_consensus_reproduces_the_published_lr_list`, which needs no expression data
    and does not depend on the stored objectives at all.
    '''
    rnaseq = c2c.io.load_rnaseq(rnaseq_file=CELEGANS_BASE + 'RNA-Seq/Celegans_RNASeqData_Cell.xlsx',
                                gene_column='symbol', drop_nangenes=True,
                                log_transformation=False, format='auto', verbose=False)
    curated = c2c.io.load_table(CELEGANS_BASE + 'PPI-Networks/Celegans-Curated-LR-pairs.xlsx',
                                format='auto', verbose=False)
    curated = remove_ppi_bidirectionality(curated, CELEGANS_COLUMNS, verbose=False)
    pool = c2c.preprocessing.preprocess_ppi_data(
        ppi_data=curated, interaction_columns=CELEGANS_COLUMNS,
        rnaseq_genes=list(rnaseq.index), upper_letter_comparison=False, verbose=False)
    distances = pd.read_csv(CELEGANS_BASE + 'Digital-3D-Map/Celegans_Physical_Distances_Min.csv',
                            index_col=0)

    listing = json.loads(urllib.request.urlopen(
        'https://api.github.com/repos/LewisLabUCSD/Celegans-cell2cell/contents/'
        'data/GA-Bray-Curtis').read())
    executions = [json.loads(urllib.request.urlopen(
        CELEGANS_BASE + 'GA-Bray-Curtis/' + entry['name']).read())
        for entry in sorted(listing, key=lambda e: e['name'])[:2]]

    setup = {'communication_score': 'expression_thresholding', 'cci_score': 'bray_curtis',
             'cci_type': 'undirected'}
    cutoffs = {'type': 'constant_value', 'parameter': 10}     # 10 TPM, as published
    vectorized = c2c.analysis.CorrelationObjective(
        rnaseq_data=rnaseq, reference_distances=distances, cutoff_setup=cutoffs,
        analysis_setup=setup, fast=True, validate_fast=False)(pool)
    reference = c2c.analysis.CorrelationObjective(
        rnaseq_data=rnaseq, reference_distances=distances, cutoff_setup=cutoffs,
        analysis_setup=setup, fast=False, validate_fast=False)(pool)

    for execution in executions:
        for run in sorted(execution, key=lambda k: int(k[3:])):
            mask = np.asarray(execution[run]['ppi_data'], dtype=float)
            assert len(mask) == len(pool)
            fast_value = vectorized(mask[None, :])[0]
            assert np.isclose(fast_value, reference.reference_value(mask), rtol=0, atol=1e-9)
            assert 0.0 <= fast_value <= 1.0


# ---------------------------------------------------------------------------------
# The pyevolve-equivalent knobs
# ---------------------------------------------------------------------------------

def test_flip_mutation_inverts_genes():
    '''`pygad`'s random mutation re-draws a gene; this one has to invert it.'''
    class _Instance:
        mutation_probability = 1.0

    offspring = np.array([[0, 1, 0, 1], [1, 1, 0, 0]])
    flipped = ga.search._flip_mutation(offspring.copy(), _Instance())
    np.testing.assert_array_equal(flipped, 1 - offspring)

    _Instance.mutation_probability = 0.0
    untouched = ga.search._flip_mutation(offspring.copy(), _Instance())
    np.testing.assert_array_equal(untouched, offspring)


def test_flip_mutation_rate_is_the_probability_asked_for(ga_inputs):
    '''Half the mutations of a re-draw are no-ops on a binary genome; none of these are.'''
    class _Instance:
        mutation_probability = 0.25

    rng = np.random.default_rng(0)
    offspring = rng.integers(0, 2, size=(200, 100))
    changed = (ga.search._flip_mutation(offspring.copy(), _Instance()) != offspring).mean()
    assert 0.2 < changed < 0.3


def test_the_ga_receives_the_pyevolve_equivalent_parameters(ga_inputs, monkeypatch):
    '''The exposed knobs have to reach `pygad`, not just sit in the signature.'''
    import inspect
    _, ppi, _ = ga_inputs
    seen = {}

    # Single-point crossover is left to `pygad`, and the equivalence with pyevolve's
    # own default rests on it staying that way. Read before the patch below replaces
    # the class with one whose signature is (*args, **kwargs).
    assert inspect.signature(pygad.GA.__init__).parameters['crossover_type'].default \
        == 'single_point'

    # A subclass, so `pygad`'s own reads of its class attributes keep working
    class Spy(pygad.GA):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pygad, 'GA', Spy)
    c2c.analysis.optimize_lr_pairs(ppi_data=ppi, objective=_RecordingObjective(),
                                   population_size=8, generations=2, runs=1,
                                   mutation_probability=0.02, crossover_probability=0.9,
                                   tournament_size=2, random_state=1)
    assert seen['mutation_probability'] == 0.02
    assert seen['crossover_probability'] == 0.9
    assert seen['K_tournament'] == 2
