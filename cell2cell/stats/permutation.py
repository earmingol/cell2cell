# -*- coding: utf-8 -*-

from __future__ import absolute_import

import scipy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import cell2cell.core.interaction_space as ispace
from cell2cell.preprocessing import shuffle_rows_in_df

from sklearn.utils import shuffle
from tqdm.auto import tqdm


def compute_pvalue_from_dist(obs_value, dist, consider_size=False, comparison='upper'):
    '''
    Computes the probability of observing a value in a given distribution.

    Parameters
    ----------
    obs_value : float
        An observed value used to get a p-value from a distribution.

    dist : array-like
        A simulated oe empirical distribution of values used to compare
        the observed value and get a p-value.

    consider_size : boolean, default=False
        Whether considering the size of the distribution for limiting
        small probabilities to be as minimal as the reciprocal of the size.

    comparison : str, default='upper'
        Type of hypothesis testing:

        - 'lower' : Lower-tailed, whether the value is smaller than most
            of the values in the distribution.
        - 'upper' : Upper-tailed, whether the value is greater than most
            of the values in the distribution.
        - 'different' : Two-tailed, whether the value is different than
            most of the values in the distribution.

    Returns
    -------
    pval : float
        P-value obtained from comparing the observed value and values in the
        distribution.
    '''
    # Omit nan values
    dist_ = [x for x in dist if ~np.isnan(x)]

    # All values in dist are NaNs or obs_value is NaN
    if (len(dist_) == 0) | np.isnan(obs_value):
        return 1.0

    # No NaN values
    if comparison == 'lower':
        pval = scipy.stats.percentileofscore(dist_, obs_value) / 100.0
    elif comparison == 'upper':
        pval = 1.0 - scipy.stats.percentileofscore(dist_, obs_value) / 100.0
    elif comparison == 'different':
        percentile = scipy.stats.percentileofscore(dist_, obs_value) / 100.0
        if percentile <= 0.5:
            pval = 2.0 * percentile
        else:
            pval = 2.0 * (1.0 - percentile)
    else:
        raise NotImplementedError('Comparison {} is not implemented'.format(comparison))

    if (consider_size) & (pval == 0.):
        pval = 1./(len(dist_) + 1e-6)

    return pval


def pvalue_from_dist(obs_value, dist, label='', consider_size=False, comparison='upper'):
    '''
    Computes a p-value for an observed value given a simulated or
    empirical distribution. It plots the distribution and prints
    the p-value.

    Parameters
    ----------
    obs_value : float
        An observed value used to get a p-value from a distribution.

    dist : array-like
        A simulated oe empirical distribution of values used to compare
        the observed value and get a p-value.

    label : str, default=''
        Label used for the histogram plot. Useful for identifying it
        across multiple plots.

    consider_size : boolean, default=False
        Whether considering the size of the distribution for limiting
        small probabilities to be as minimal as the reciprocal of the size.

    comparison : str, default='upper'
        Type of hypothesis testing:

        - 'lower' : Lower-tailed, whether the value is smaller than most
            of the values in the distribution.
        - 'upper' : Upper-tailed, whether the value is greater than most
            of the values in the distribution.
        - 'different' : Two-tailed, whether the value is different than
            most of the values in the distribution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure that shows the histogram for dist.

    pval : float
        P-value obtained from comparing the observed value and values in the
        distribution.
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


def random_switching_ppi_labels(ppi_data, genes=None, random_state=None, interaction_columns=('A', 'B'), permuted_column='both'):
    '''
    Randomly permutes the labels of interacting proteins in
    a list of protein-protein interactions.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    genes : list, default=None
        List of genes, with names matching proteins in the PPIs, to exclusively
        consider in the analysis.

    random_state : int, default=None
        Seed for randomization.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    permuted_column : str, default='both'
        Column among the interacting_columns to permute.
        Options are:

        - 'first' : To permute labels considering only proteins in the first
            column.
        - 'second' : To permute labels considering only proteins in the second
            column.
        - ' both' : To permute labels considering all the proteins in the list.

    Returns
    -------
    ppi_data_ : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) with randomly
        permuted labels of proteins.
    '''
    ppi_data_ = ppi_data.copy()
    prot_a = interaction_columns[0]
    prot_b = interaction_columns[1]
    if permuted_column == 'both':
        if genes is None:
            genes = list(np.unique(ppi_data_[interaction_columns].values.flatten()))
        else:
            genes = list(set(genes))
        mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
        ppi_data_[prot_a] = ppi_data_[prot_a].apply(lambda x: mapper[x])
        ppi_data_[prot_b] = ppi_data_[prot_b].apply(lambda x: mapper[x])
    elif permuted_column == 'first':
        if genes is None:
            genes = list(np.unique(ppi_data_[prot_a].values.flatten()))
        else:
            genes = list(set(genes))
        mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
        ppi_data_[prot_a] = ppi_data_[prot_a].apply(lambda x: mapper[x])
    elif permuted_column == 'second':
        if genes is None:
            genes = list(np.unique(ppi_data_[prot_b].values.flatten()))
        else:
            genes = list(set(genes))
        mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
        ppi_data_[prot_b] = ppi_data_[prot_b].apply(lambda x: mapper[x])
    else: raise ValueError('Not valid option')
    return ppi_data_


def run_label_permutation(rnaseq_data, ppi_data, genes, analysis_setup, cutoff_setup, permutations=10000,
                          permuted_label='gene_labels', excluded_cells=None, consider_size=True, verbose=False):
    '''Permutes a label before computing cell-cell interaction scores.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment or a single-cell
        experiment after aggregation into cell types. Columns are
        cell-types/tissues/samples and rows are genes.

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    genes : list
        List of genes in rnaseq_data to exclusively consider in the analysis.

    analysis_setup : dict
        Contains main setup for running the cell-cell interactions and communication
        analyses. Three main setups are needed (passed as keys):

        - 'communication_score' : is the type of communication score used to detect
            active ligand-receptor pairs between each pair of cell.
            It can be:

            - 'expression_thresholding'
            - 'expression_product'
            - 'expression_mean'
        - 'cci_score' : is the scoring function to aggregate the communication
            scores.
            It can be:

            - 'bray_curtis'
            - 'jaccard'
            - 'count'
        - 'cci_type' : is the type of interaction between two cells. If it is
            undirected, all ligands and receptors are considered from both cells.
            If it is directed, ligands from one cell and receptors from the other
             are considered separately with respect to ligands from the second
             cell and receptor from the first one.
             So, it can be:

             - 'undirected'
             - 'directed

    cutoff_setup : dict
        Contains two keys: 'type' and 'parameter'. The first key represent the
        way to use a cutoff or threshold, while parameter is the value used
        to binarize the expression values.
        The key 'type' can be:

        - 'local_percentile' : computes the value of a given percentile, for each
            gene independently. In this case, the parameter corresponds to the
            percentile to compute, as a float value between 0 and 1.
        - 'global_percentile' : computes the value of a given percentile from all
            genes and samples simultaneously. In this case, the parameter
            corresponds to the percentile to compute, as a float value between
            0 and 1. All genes have the same cutoff.
        - 'file' : load a cutoff table from a file. Parameter in this case is the
            path of that file. It must contain the same genes as index and same
            samples as columns.
        - 'multi_col_matrix' : a dataframe must be provided, containing a cutoff
            for each gene in each sample. This allows to use specific cutoffs for
            each sample. The columns here must be the same as the ones in the
            rnaseq_data.
        - 'single_col_matrix' : a dataframe must be provided, containing a cutoff
            for each gene in only one column. These cutoffs will be applied to
            all samples.
        - 'constant_value' : binarizes the expression. Evaluates whether
            expression is greater than the value input in the parameter.

    permutations : int, default=100
            Number of permutations where in each of them a random
            shuffle of labels is performed, followed of computing
            CCI scores to create a null distribution.

    permuted_label : str, default='gene_labels'
        Label to be permuted.
        Types are:

            - 'genes' : Permutes cell-labels in a gene-specific way.
            - 'gene_labels' : Permutes the labels of genes in the RNA-seq dataset.
            - 'cell_labels' : Permutes the labels of cell-types/tissues/samples
                in the RNA-seq dataset.

    excluded_cells : list, default=None
        List of cells to exclude from the analysis.

    consider_size : boolean, default=True
        Whether considering the size of the distribution for limiting
        small probabilities to be as minimal as the reciprocal of the size.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    Returns
    -------
    cci_pvals : pandas.DataFrame
        Matrix where rows and columns are cell-types/tissues/samples and
        each value is a P-value for the corresponding CCI score.
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

        if permuted_label == 'genes':
            # Shuffle genes
            shuffled_rnaseq_data = shuffle_rows_in_df(df=rnaseq_data_,
                                                      rows=genes)
        elif permuted_label == 'cell_labels':
            # Shuffle cell labels
            shuffled_rnaseq_data = rnaseq_data_.copy()
            shuffled_cells = shuffle(list(shuffled_rnaseq_data.columns))
            shuffled_rnaseq_data.columns = shuffled_cells

        elif permuted_label == 'gene_labels':
            # Shuffle gene labels
            shuffled_rnaseq_data = rnaseq_data.copy()
            shuffled_genes = shuffle(list(shuffled_rnaseq_data.index))
            shuffled_rnaseq_data.index = shuffled_genes
        else:
            raise ValueError('Not a valid shuffle_type.')


        interaction_space = ispace.InteractionSpace(rnaseq_data=shuffled_rnaseq_data.loc[genes, included_cells],
                                                    ppi_data=ppi_data,
                                                    gene_cutoffs=cutoff_setup,
                                                    communication_score=analysis_setup['communication_score'],
                                                    cci_score=analysis_setup['cci_score'],
                                                    cci_type=analysis_setup['cci_type'],
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
    base_interaction_space = ispace.InteractionSpace(rnaseq_data=rnaseq_data_[included_cells],
                                                     ppi_data=ppi_data,
                                                     gene_cutoffs=cutoff_setup,
                                                     communication_score=analysis_setup['communication_score'],
                                                     cci_score=analysis_setup['cci_score'],
                                                     cci_type=analysis_setup['cci_type'],
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