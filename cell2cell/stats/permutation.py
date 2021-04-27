# -*- coding: utf-8 -*-

from __future__ import absolute_import

import scipy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from cell2cell.analysis import initialize_interaction_space
from cell2cell.preprocessing import shuffle_rows_in_df

from sklearn.utils import shuffle
from tqdm.auto import tqdm

def compute_pvalue_from_dist(obs_value, dist, consider_size=False, comparison='upper'):
    if comparison == 'lower':
        pval = scipy.stats.percentileofscore(dist, obs_value) / 100.0
    elif comparison == 'upper':
        pval = 1.0 - scipy.stats.percentileofscore(dist, obs_value) / 100.0
    elif comparison == 'different':
        percentile = scipy.stats.percentileofscore(dist, obs_value) / 100.0
        if percentile <= 0.5:
            pval = 2.0 * percentile
        else:
            pval = 2.0 * (1.0 - percentile)
    else:
        raise NotImplementedError('Comparison {} is not implemented'.format(comparison))

    if (consider_size) & (pval == 0.):
        pval = 1./(len(dist) + 1e-6)

    return pval


def pvalue_from_dist(obs_value, dist, label='', consider_size=False, comparison='upper'):
    '''
    This function computes a p-value for an observed value given a simulated or empirical distribution.
    It plots the distribution and prints the p-value.

    Parameters
    ----------
    obs_value : float
        Observed value used to get a p-value from a distribution.

    dist : array-like
        Simulated/Empirical distribution used to compare the observed value and get a p-value

    label : str, '' by default
        Label used for the histogram plot.

    consider_size : boolean, False by default
        Consider size of distribution for limiting the p-value to be as minimal as the reciprocal of the size.

    comparison : str, 'upper' by default
        Type of comparison used to get the p-value. It could be 'lower', 'upper' or 'different'.
        Lower and upper correspond to one-tailed analysis, while different is for a two-tailed analysis.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure that shows the histogram for dist.

    pval : float
        P-value obtained from both observed value and pre-computed distribution.
    '''
    pval = compute_pvalue_from_dist(obs_value=obs_value,
                                    dist=dist,
                                    consider_size=consider_size,
                                    comparison=comparison
                                    )

    print('P-value is: {}'.format(pval))

    with sns.axes_style("darkgrid"):
        if label == '':
            if pval > 1. - 1. / len(dist):
                label_ = 'p-val: >{:g}'.format(float('{:.1g}'.format(1. - 1. / len(dist))))
            elif pval < 1. / len(dist):
                label_ = 'p-val: <{:g}'.format(float('{:.1g}'.format(1. / len(dist))))
            else:
                label_ = 'p-val: {0:.2E}'.format(pval)
        else:
            if pval > 1. - 1. / len(dist):
                label_ = label + ' - p-val: >{:g}'.format(float('{:.1g}'.format(1. - 1. / len(dist))))
            elif pval < 1. / len(dist):
                label_ = label + ' - p-val: <{:g}'.format(float('{:.1g}'.format(1. / len(dist))))
            else:
                label_ = label + ' - p-val: {0:.2E}'.format(pval)
        fig = sns.distplot(dist, hist=True, kde=True, norm_hist=False, rug=False, label=label_)
        fig.axvline(x=obs_value, color=fig.get_lines()[-1].get_c(), ls='--')

        fig.tick_params(axis='both', which='major', labelsize=16)
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                     ncol=1, fancybox=True, shadow=True, fontsize=14)
    return fig, pval


def random_switching_ppi_labels(ppi_data, genes=None, random_state=None, interaction_columns=('A', 'B'), column='both'):
    ppi_data_ = ppi_data.copy()
    prot_a = interaction_columns[0]
    prot_b = interaction_columns[1]
    if column == 'both':
        if genes is None:
            genes = list(np.unique(ppi_data_[interaction_columns].values.flatten()))
        else:
            genes = list(set(genes))
        mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
        ppi_data_[prot_a] = ppi_data_[prot_a].apply(lambda x: mapper[x])
        ppi_data_[prot_b] = ppi_data_[prot_b].apply(lambda x: mapper[x])
    elif column == 'first':
        if genes is None:
            genes = list(np.unique(ppi_data_[prot_a].values.flatten()))
        else:
            genes = list(set(genes))
        mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
        ppi_data_[prot_a] = ppi_data_[prot_a].apply(lambda x: mapper[x])
    elif column == 'second':
        if genes is None:
            genes = list(np.unique(ppi_data_[prot_b].values.flatten()))
        else:
            genes = list(set(genes))
        mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
        ppi_data_[prot_b] = ppi_data_[prot_b].apply(lambda x: mapper[x])
    else: raise ValueError('Not valid option')
    return ppi_data_


def run_cellwise_permutation(rnaseq_data, ppi_data, genes, cutoff_setup, analysis_setup, permutations=10000, shuffle_type='gene_labels',
                             excluded_cells=None, use_ppi_score=False, consider_size=True, verbose = False):
    '''
    shuffle_type : 'genes, gene_labels, cell_labels
    '''
    # Placeholders
    scores = np.array([])
    diag = np.array([])

    # Info to use
    if genes is None:
        genes = list(rnaseq_data.index)

    if excluded_cells is not None:
        included_cells = sorted(list(set(rnaseq_data.columns) - set(excluded_cells)))
    else:
        included_cells = sorted(list(set(rnaseq_data.columns)))

    rnaseq_data_ = rnaseq_data.loc[genes, included_cells]

    # Permutations
    for i in tqdm(range(permutations)):

        if shuffle_type == 'genes':
            # Shuffle genes
            shuffled_rnaseq_data = shuffle_rows_in_df(df=rnaseq_data_,
                                                      rows=genes)
        elif shuffle_type == 'cell_labels':
            # Shuffle cell labels
            shuffled_rnaseq_data = rnaseq_data_.copy()
            shuffled_cells = shuffle(list(shuffled_rnaseq_data.columns))
            shuffled_rnaseq_data.columns = shuffled_cells

        elif shuffle_type == 'gene_labels':
            # Shuffle gene labels
            shuffled_rnaseq_data = rnaseq_data.copy()
            shuffled_genes = shuffle(list(shuffled_rnaseq_data.index))
            shuffled_rnaseq_data.index = shuffled_genes
        else:
            raise ValueError('Not a valid shuffle_type.')

        interaction_space = initialize_interaction_space(rnaseq_data=shuffled_rnaseq_data.loc[genes, included_cells],
                                                         ppi_data=ppi_data,
                                                         cutoff_setup=cutoff_setup,
                                                         analysis_setup=analysis_setup,
                                                         excluded_cells=excluded_cells,
                                                         use_ppi_score=use_ppi_score,
                                                         verbose=verbose)

        # Keep scores
        cci = interaction_space.interaction_elements['cci_matrix'].loc[included_cells, included_cells]
        cci_diag = np.diag(cci).copy()
        np.fill_diagonal(cci.values, 0.0)

        iter_scores = scipy.spatial.distance.squareform(cci)
        iter_scores = np.reshape(iter_scores, (len(iter_scores), 1)).T

        iter_diag = np.reshape(cci_diag, (len(cci_diag), 1)).T

        if scores.shape == (0,):
            scores = iter_scores
        else:
            scores = np.concatenate([scores, iter_scores], axis=0)

        if diag.shape == (0,):
            diag = iter_diag
        else:
            diag = np.concatenate([diag, iter_diag], axis=0)

    # Base CCI scores
    base_interaction_space = initialize_interaction_space(rnaseq_data=rnaseq_data_,
                                                          ppi_data=ppi_data,
                                                          cutoff_setup=cutoff_setup,
                                                          analysis_setup=analysis_setup,
                                                          excluded_cells=excluded_cells,
                                                          use_ppi_score=use_ppi_score,
                                                          verbose=verbose)

    # Keep scores
    base_cci = base_interaction_space.interaction_elements['cci_matrix'].loc[included_cells, included_cells]
    base_cci_diag = np.diag(base_cci).copy()
    np.fill_diagonal(base_cci.values, 0.0)

    base_scores = scipy.spatial.distance.squareform(base_cci)

    # P-values
    pvals = np.zeros((scores.shape[1], 1))
    diag_pvals = np.zeros((diag.shape[1], 1))

    for i in range(scores.shape[1]):
        pvals[i] = compute_pvalue_from_dist(base_scores[i],
                                            scores[:, i],
                                            consider_size=consider_size,
                                            comparison='different'
                                            )

    for i in range(diag.shape[1]):
        diag_pvals[i] = compute_pvalue_from_dist(base_cci_diag[i],
                                                 diag[:, i],
                                                 consider_size=consider_size,
                                                 comparison='different'
                                                 )

    # DataFrame
    cci_pvals = scipy.spatial.distance.squareform(pvals.reshape((len(pvals),)))
    for i, v in enumerate(diag_pvals.flatten()):
        cci_pvals[i, i] = v
    cci_pvals = pd.DataFrame(cci_pvals, index=base_cci.index, columns=base_cci.columns)

    return cci_pvals