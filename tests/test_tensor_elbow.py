# -*- coding: utf-8 -*-

'''Tests for the elbow analysis: cell2cell.tensor.factorization and its plots.

These exercise the rank-selection workflow, which is how users choose the number of
factors, and the plotting helpers that report it.
'''

import numpy as np
import pytest
from matplotlib import pyplot as plt

import cell2cell as c2c
from cell2cell.plotting import tensor_plot
from cell2cell.tensor import factorization


# ---------------------------------------------------------------------------------
# elbow_rank_selection
# ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_elbow_rank_selection_single_run(interaction_tensor):
    fig, errors = interaction_tensor.elbow_rank_selection(
        upper_rank=4, runs=1, automatic_elbow=False, manual_elbow=2,
        random_state=0, verbose=False)
    assert len(errors) == 4
    assert interaction_tensor.rank == 2
    # Errors are (rank, error) pairs with decreasing rank error overall
    ranks = [rank for rank, _ in errors]
    assert ranks == [1, 2, 3, 4]


@pytest.mark.slow
def test_elbow_rank_selection_multiple_runs(interaction_tensor):
    fig, errors = interaction_tensor.elbow_rank_selection(
        upper_rank=3, runs=2, automatic_elbow=False, manual_elbow=2,
        random_state=0, verbose=False)
    assert len(errors) == 3
    assert interaction_tensor.rank == 2
    for entry in errors:
        assert len(entry) == 2


@pytest.mark.slow
def test_elbow_rank_selection_automatic_needs_a_detectable_elbow(interaction_tensor):
    '''Documents a limitation: when the elbow detector finds no elbow it returns None,
    and the code then calls int(None), raising a TypeError instead of a clear message.
    The toy tensor is small enough to hit that path.
    '''
    with pytest.raises(TypeError):
        interaction_tensor.elbow_rank_selection(upper_rank=5, runs=1,
                                                automatic_elbow=True,
                                                random_state=0, verbose=False)


@pytest.mark.slow
def test_elbow_rank_selection_is_reproducible(interaction_tensor):
    _, first = interaction_tensor.elbow_rank_selection(
        upper_rank=3, runs=1, automatic_elbow=False, manual_elbow=2,
        random_state=5, verbose=False)
    _, second = interaction_tensor.elbow_rank_selection(
        upper_rank=3, runs=1, automatic_elbow=False, manual_elbow=2,
        random_state=5, verbose=False)
    assert np.allclose([e for _, e in first], [e for _, e in second])


@pytest.mark.slow
def test_elbow_rank_selection_without_a_figure(interaction_tensor):
    result = interaction_tensor.elbow_rank_selection(
        upper_rank=3, runs=1, automatic_elbow=False, manual_elbow=2,
        random_state=0, output_fig=False, verbose=False)
    fig, errors = result
    assert fig is None
    assert len(errors) == 3


@pytest.mark.slow
def test_elbow_rank_selection_smoothing_needs_enough_ranks(interaction_tensor):
    '''smooth=True runs a Savitzky-Golay filter, whose polynomial order must be
    smaller than the window, so it needs a reasonable number of ranks.'''
    with pytest.raises(Exception):
        interaction_tensor.elbow_rank_selection(upper_rank=4, runs=1,
                                                automatic_elbow=True, smooth=True,
                                                random_state=0, verbose=False)


# ---------------------------------------------------------------------------------
# Factorization internals
# ---------------------------------------------------------------------------------

def test_compute_tensor_factorization_returns_errors(interaction_tensor):
    interaction_tensor.compute_tensor_factorization(rank=2, random_state=0)
    errors = interaction_tensor.get_factorization_errors()
    assert errors is not None
    assert len(errors) > 0


def test_factorization_errors_decrease(interaction_tensor):
    interaction_tensor.compute_tensor_factorization(rank=2, random_state=0)
    errors = np.asarray(interaction_tensor.get_factorization_errors())
    # The optimization must not make the reconstruction worse overall
    assert errors[-1] <= errors[0] + 1e-9


@pytest.mark.parametrize('init', ['random', 'svd'])
def test_factorization_initializations(interaction_tensor, init):
    interaction_tensor.compute_tensor_factorization(rank=2, init=init, random_state=0)
    assert len(interaction_tensor.factors) == 4


def test_factorization_without_normalizing_loadings(interaction_tensor):
    interaction_tensor.compute_tensor_factorization(rank=2, random_state=0,
                                                    normalize_loadings=False)
    assert interaction_tensor.explained_variance_ratio_ is None
    assert len(interaction_tensor.factors) == 4


def test_compute_elbow_picks_a_rank():
    loss = [(1, 0.9), (2, 0.5), (3, 0.45), (4, 0.44), (5, 0.43)]
    elbow = factorization._compute_elbow(loss)
    assert 1 <= elbow <= 5


def test_compute_norm_error_is_between_zero_and_one(factorized_tensor):
    '''Signature is (tensor, tl_object), in that order.'''
    error = factorization._compute_norm_error(factorized_tensor.tensor,
                                              factorized_tensor.tl_object)
    assert 0.0 <= error <= 1.0


# ---------------------------------------------------------------------------------
# Elbow plots
# ---------------------------------------------------------------------------------

def test_plot_elbow_returns_a_figure():
    loss = [(1, 0.9), (2, 0.5), (3, 0.4), (4, 0.38)]
    fig = tensor_plot.plot_elbow(loss, elbow=2)
    assert fig is not None


def test_plot_elbow_without_an_elbow():
    loss = [(1, 0.9), (2, 0.5), (3, 0.4)]
    assert tensor_plot.plot_elbow(loss) is not None


def test_plot_multiple_run_elbow():
    all_loss = np.array([[0.9, 0.85, 0.88],
                         [0.5, 0.52, 0.49],
                         [0.4, 0.41, 0.39]]).T
    fig = tensor_plot.plot_multiple_run_elbow(all_loss, elbow=2)
    assert fig is not None


def test_plot_multiple_run_elbow_with_std():
    all_loss = np.array([[0.9, 0.85, 0.88],
                         [0.5, 0.52, 0.49],
                         [0.4, 0.41, 0.39]]).T
    fig = tensor_plot.plot_multiple_run_elbow(all_loss, elbow=2, ci='std')
    assert fig is not None


def test_plot_factorization_errors():
    errors = [0.9, 0.6, 0.5, 0.45, 0.44]
    fig = tensor_plot.plot_factorization_errors(errors)
    assert fig is not None


def test_generate_plot_df(factorized_tensor):
    frame = tensor_plot.generate_plot_df(factorized_tensor)
    assert frame is not None
    assert frame.shape[0] > 0


# ---------------------------------------------------------------------------------
# The masked path
#
# A mask marks values that should not count towards the reconstruction error. It
# forces `init='random'` and routes the error through the masked branch of
# `_compute_norm_error`, which is a different formula from the unmasked one.
# ---------------------------------------------------------------------------------

@pytest.fixture
def masked_tensor():
    '''A small PreBuiltTensor with one missing value, and the matching mask.'''
    shape = (2, 3, 3, 3)
    values = (np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.) / np.prod(shape)
    values[0, 0, 0, 0] = np.nan
    mask = (~np.isnan(values)).astype(int)
    names = [['Context-1', 'Context-2'],
             ['LR-1', 'LR-2', 'LR-3'],
             ['C1', 'C2', 'C3'],
             ['C1', 'C2', 'C3']]
    return c2c.tensor.PreBuiltTensor(tensor=values, order_names=names, mask=mask,
                                     order_labels=['Contexts', 'Ligand-Receptor Pairs',
                                                   'Sender Cells', 'Receiver Cells'])


def test_masked_tensor_keeps_its_mask(masked_tensor):
    assert masked_tensor.mask is not None
    assert masked_tensor.missing_fraction() > 0.


def test_compute_norm_error_ignores_the_masked_entries(masked_tensor):
    '''The masked error only compares the entries the mask keeps, so it differs from
    the error computed over the whole tensor.'''
    masked_tensor.compute_tensor_factorization(rank=2, random_state=0, n_iter_max=20)
    with_mask = factorization._compute_norm_error(masked_tensor.tensor,
                                                  masked_tensor.tl_object,
                                                  masked_tensor.mask)
    without_mask = factorization._compute_norm_error(masked_tensor.tensor,
                                                     masked_tensor.tl_object)
    assert 0.0 <= float(with_mask) <= 1.0
    assert not np.isclose(float(with_mask), float(without_mask))


def test_compute_norm_error_needs_a_factorization():
    with pytest.raises(AssertionError):
        factorization._compute_norm_error(np.ones((2, 2, 2)), None)


@pytest.mark.slow
def test_elbow_analysis_with_a_mask(masked_tensor):
    '''The mask is picked up from the tensor and used for every rank.'''
    fig, errors = masked_tensor.elbow_rank_selection(
        upper_rank=2, runs=1, automatic_elbow=False, manual_elbow=1,
        n_iter_max=10, tol=1e-3, random_state=0, verbose=False)
    assert len(errors) == 2
    assert all(0.0 <= float(error) <= 1.0 for _, error in errors)


@pytest.mark.slow
def test_multiple_runs_elbow_analysis_with_a_mask(masked_tensor):
    all_loss = factorization._multiple_runs_elbow_analysis(
        masked_tensor.tensor, upper_rank=2, runs=2, mask=masked_tensor.mask,
        n_iter_max=10, tol=1e-3, random_state=0)
    assert all_loss.shape == (2, 2)


# ---------------------------------------------------------------------------------
# metric='similarity'
#
# Instead of the reconstruction error, the runs of each rank are compared to each
# other with the correlation index, so a rank is good when its runs agree.
# ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_multiple_runs_elbow_analysis_with_the_similarity_metric(interaction_tensor):
    all_loss = factorization._multiple_runs_elbow_analysis(
        interaction_tensor.tensor, upper_rank=2, runs=2, metric='similarity',
        n_iter_max=10, tol=1e-3, random_state=0)
    # With `runs` runs there are runs*(runs-1)/2 pairwise comparisons per rank
    assert all_loss.shape == (1, 2)
    assert np.all(np.isfinite(all_loss))


@pytest.mark.slow
def test_elbow_rank_selection_with_the_similarity_metric(interaction_tensor):
    fig, errors = interaction_tensor.elbow_rank_selection(
        upper_rank=2, runs=2, metric='similarity', automatic_elbow=False,
        manual_elbow=1, n_iter_max=10, tol=1e-3, random_state=0, verbose=False)
    assert interaction_tensor.elbow_metric == 'similarity'
    assert len(errors) == 2
