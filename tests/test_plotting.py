# -*- coding: utf-8 -*-

'''Smoke tests for cell2cell.plotting.

Plots are only checked for running without error and returning a figure/axes; no
image comparison is done. The Agg backend and the figure teardown come from conftest.
'''

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import cell2cell as c2c
from cell2cell.analysis.tensor_downstream import get_factor_specific_ccc_networks


# ---------------------------------------------------------------------------------
# aesthetics
# ---------------------------------------------------------------------------------

def test_get_colors_from_labels():
    labels = ['a', 'b', 'c']
    colors = c2c.plotting.get_colors_from_labels(labels)
    assert set(colors.keys()) == set(labels)
    for value in colors.values():
        assert len(value) == 4          # RGBA


def test_get_colors_from_labels_are_distinct():
    colors = c2c.plotting.get_colors_from_labels(['a', 'b', 'c', 'd'])
    assert len({tuple(v) for v in colors.values()}) == 4


def test_get_colors_from_labels_single_label():
    colors = c2c.plotting.get_colors_from_labels(['only'])
    assert len(colors) == 1


def test_map_colors_to_metadata(toy_metadata, toy_rnaseq):
    colors = c2c.plotting.map_colors_to_metadata(metadata=toy_metadata,
                                                 ref_df=toy_rnaseq,
                                                 sample_col='#SampleID',
                                                 group_col='Groups')
    assert len(colors) == toy_rnaseq.shape[1]


def test_generate_legend_returns_a_legend():
    colors = c2c.plotting.get_colors_from_labels(['a', 'b'])
    fig, ax = plt.subplots()
    legend = c2c.plotting.generate_legend(colors, ax=ax)
    assert legend is not None


def test_generate_legend_sorted_labels_are_natural():
    colors = c2c.plotting.get_colors_from_labels(['CT-10', 'CT-2', 'CT-1'])
    fig, ax = plt.subplots()
    legend = c2c.plotting.generate_legend(colors, ax=ax, sorted_labels=True)
    texts = [t.get_text() for t in legend.get_texts()]
    assert texts == ['CT-1', 'CT-2', 'CT-10']


# ---------------------------------------------------------------------------------
# clustermaps
# ---------------------------------------------------------------------------------

def test_clustermap_cci(bulk_interactions, toy_metadata):
    grid = c2c.plotting.clustermap_cci(bulk_interactions, metadata=toy_metadata,
                                       sample_col='#SampleID', group_col='Groups')
    assert grid is not None


def test_clustermap_cci_without_metadata(bulk_interactions):
    grid = c2c.plotting.clustermap_cci(bulk_interactions)
    assert grid is not None


def test_clustermap_ccc(bulk_interactions, toy_metadata):
    grid = c2c.plotting.clustermap_ccc(bulk_interactions, metadata=toy_metadata,
                                       sample_col='#SampleID', group_col='Groups')
    assert grid is not None


# ---------------------------------------------------------------------------------
# circos
# ---------------------------------------------------------------------------------

def test_circos_plot(bulk_interactions, toy_metadata):
    result = c2c.plotting.circos_plot(interaction_space=bulk_interactions,
                                      sender_cells=['C1', 'C2'],
                                      receiver_cells=['C3', 'C4'],
                                      ligands=['Protein-A'],
                                      receptors=['Protein-B'],
                                      metadata=toy_metadata,
                                      sample_col='#SampleID', group_col='Groups',
                                      excluded_score=-1)
    assert result is not None


# ---------------------------------------------------------------------------------
# factor plots
# ---------------------------------------------------------------------------------

def test_ccc_networks_plot(factorized_tensor):
    fig, axes = c2c.plotting.ccc_networks_plot(factorized_tensor.factors,
                                               included_factors=['Factor 1', 'Factor 2'],
                                               ccc_threshold=0.01, nrows=1,
                                               panel_size=(3, 3))
    assert fig is not None
    assert len(np.ravel(axes)) >= 2


def test_ccc_networks_plot_all_factors(factorized_tensor):
    fig, axes = c2c.plotting.ccc_networks_plot(factorized_tensor.factors,
                                               panel_size=(3, 3))
    assert fig is not None


def test_context_boxplot(factorized_tensor):
    contexts = list(factorized_tensor.order_names[0])
    metadict = {name: ('early' if i % 2 else 'late')
                for i, name in enumerate(contexts)}
    fig, axes = c2c.plotting.context_boxplot(factorized_tensor.factors['Contexts'],
                                             metadict=metadict, nrows=1,
                                             statistical_test=None)
    assert fig is not None


def test_context_boxplot_group_order_is_natural(factorized_tensor):
    contexts = list(factorized_tensor.order_names[0])
    metadict = {name: 'G-{}'.format((i % 3) + 1) for i, name in enumerate(contexts)}
    fig, axes = c2c.plotting.context_boxplot(factorized_tensor.factors['Contexts'],
                                             metadict=metadict, nrows=1,
                                             statistical_test=None)
    assert fig is not None


def test_loading_clustermap(factorized_tensor):
    grid = c2c.plotting.loading_clustermap(factorized_tensor.factors['Sender Cells'],
                                           use_zscore=False)
    assert grid is not None


def test_loading_clustermap_with_zscore(factorized_tensor):
    grid = c2c.plotting.loading_clustermap(factorized_tensor.factors['Ligand-Receptor Pairs'],
                                           use_zscore=True)
    assert grid is not None


# ---------------------------------------------------------------------------------
# tensor plots
# ---------------------------------------------------------------------------------

def test_tensor_factors_plot(factorized_tensor):
    fig, axes = c2c.plotting.tensor_factors_plot(factorized_tensor,
                                                 order_labels=list(factorized_tensor.factors.keys()))
    assert fig is not None


def test_tensor_factors_plot_from_loadings(factorized_tensor):
    fig, axes = c2c.plotting.tensor_factors_plot_from_loadings(
        factorized_tensor.factors,
        order_labels=list(factorized_tensor.factors.keys()))
    assert fig is not None


def test_plot_elbow():
    from cell2cell.plotting.tensor_plot import plot_elbow
    loss = [(1, 0.9), (2, 0.5), (3, 0.4), (4, 0.38)]
    fig = plot_elbow(loss, elbow=2)
    assert fig is not None


def test_reorder_dimension_elements(factorized_tensor):
    '''Returns a (reordered_factors, new_metadata) tuple; metadata may be None.

    The `metadata=None` default used to crash, because `metadata.copy()` ran
    unconditionally even though the lines after it already guard with
    `if new_metadata is not None`.
    '''
    from cell2cell.plotting.tensor_plot import reorder_dimension_elements
    cells = list(factorized_tensor.order_names[2])
    reordered, metadata = reorder_dimension_elements(factorized_tensor.factors,
                                                     {'Sender Cells': cells[::-1]})
    assert metadata is None
    assert list(reordered['Sender Cells'].index) == cells[::-1]
    # The other dimensions keep their original order
    assert list(reordered['Receiver Cells'].index) == cells


def test_reorder_dimension_elements_with_metadata(factorized_tensor):
    from cell2cell.plotting.tensor_plot import reorder_dimension_elements
    cells = list(factorized_tensor.order_names[2])
    metadata = c2c.tensor.generate_tensor_metadata(
        interaction_tensor=factorized_tensor,
        metadata_dicts=[None, None, None, None],
        fill_with_order_elements=True)
    reordered, new_metadata = reorder_dimension_elements(
        factorized_tensor.factors, {'Sender Cells': cells[::-1]}, metadata=metadata)
    assert list(reordered['Sender Cells'].index) == cells[::-1]
    assert list(new_metadata[2]['Element']) == cells[::-1]


def test_reorder_dimension_elements_rejects_unknown_dimension(factorized_tensor):
    from cell2cell.plotting.tensor_plot import reorder_dimension_elements
    with pytest.raises(AssertionError):
        reorder_dimension_elements(factorized_tensor.factors, {'Not-A-Dimension': []})


# ---------------------------------------------------------------------------------
# pcoa and umap
# ---------------------------------------------------------------------------------

def test_pcoa_3dplot(bulk_interactions, toy_metadata):
    fig = c2c.plotting.pcoa_3dplot(interaction_space=bulk_interactions,
                                   metadata=toy_metadata, sample_col='#SampleID',
                                   group_col='Groups')
    assert fig is not None


def test_umap_biplot(toy_rnaseq):
    umap_df = c2c.external.run_umap(toy_rnaseq, axis=1, n_neighbors=3, random_state=0)
    fig = c2c.plotting.umap_biplot(umap_df)
    assert fig is not None


# ---------------------------------------------------------------------------------
# dot plots
# ---------------------------------------------------------------------------------

def test_generate_dot_plot():
    index = ['LR-1', 'LR-2']
    columns = ['C1 --> C2', 'C1 --> C3']
    pvals = pd.DataFrame([[0.01, 0.2], [0.3, 0.001]], index=index, columns=columns)
    scores = pd.DataFrame([[1.0, 0.5], [0.2, 0.9]], index=index, columns=columns)
    fig = c2c.plotting.generate_dot_plot(pval_df=pvals, score_df=scores)
    assert fig is not None


def test_generate_dot_plot_sizes_the_dots_by_significance():
    '''Dot sizes come from a -log10 transform of the p-values, which used to be computed
    with the `applymap` method that pandas 3.0 removed. The other dot plot tests only
    assert that a figure comes back, so this one checks the dots that were drawn.'''
    index = ['LR-1', 'LR-2']
    columns = ['C1 --> C2', 'C1 --> C3']
    pvals = pd.DataFrame([[0.01, 0.2], [0.3, 0.001]], index=index, columns=columns)
    scores = pd.DataFrame([[1.0, 0.5], [0.2, 0.9]], index=index, columns=columns)
    fig = c2c.plotting.generate_dot_plot(pval_df=pvals, score_df=scores, significance=1.0)

    # Every dot is scattered individually, row by row, on the main (second) subplot
    main_ax = fig.axes[1]
    sizes = np.array([collection.get_sizes()[0] for collection in main_ax.collections])
    assert len(sizes) == pvals.size

    # A more significant interaction must get a strictly bigger dot
    flat_pvals = pvals.values.ravel()
    assert sizes.argmax() == flat_pvals.argmin()
    by_significance = np.argsort(flat_pvals)
    assert list(sizes[by_significance]) == sorted(sizes, reverse=True)


@pytest.mark.slow
def test_dot_plot(toy_single_cells, toy_ppi):
    rnaseq, metadata = toy_single_cells
    interactions = c2c.analysis.SingleCellInteractions(
        rnaseq_data=rnaseq, ppi_data=toy_ppi, metadata=metadata,
        barcode_col='barcodes', celltype_col='cell_types',
        communication_score='expression_product', complex_sep=None, verbose=False)
    interactions.compute_pairwise_communication_scores(verbose=False)
    interactions.permute_cell_labels(evaluation='communication', permutations=10,
                                     random_state=0, verbose=False)
    # significance=1.0 keeps every interaction; with a stricter cutoff the toy data
    # can leave nothing to plot, and the underlying plotting code cannot handle an
    # entirely empty frame.
    fig = c2c.plotting.dot_plot(interactions, evaluation='communication',
                                significance=1.0)
    assert fig is not None
