# -*- coding: utf-8 -*-

from __future__ import absolute_import

import scipy
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle


def pvalue_from_dist(obs_value, dist, label='', comparison='upper'):
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


def random_switching_ppi_labels(ppi_data, genes=None, random_state=None, interaction_columns=['A', 'B'], column='both'):
    ppi_data_ = ppi_data.copy()
    A = interaction_columns[0]
    B = interaction_columns[1]
    if column == 'both':
        if genes is None:
            genes = list(np.unique(ppi_data_[interaction_columns].values.flatten()))
        else:
            genes = list(set(genes))
        mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
        ppi_data_[A] = ppi_data_[A].apply(lambda x: mapper[x])
        ppi_data_[B] = ppi_data_[B].apply(lambda x: mapper[x])
    elif column == 'first':
        if genes is None:
            genes = list(np.unique(ppi_data_[A].values.flatten()))
        else:
            genes = list(set(genes))
        mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
        ppi_data_[A] = ppi_data_[A].apply(lambda x: mapper[x])
    elif column == 'second':
        if genes is None:
            genes = list(np.unique(ppi_data_[B].values.flatten()))
        else:
            genes = list(set(genes))
        mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
        ppi_data_[B] = ppi_data_[B].apply(lambda x: mapper[x])
    else: raise ValueError('Not valid option')
    return ppi_data_