# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd
import scanpy
import numpy as np
from tqdm import tqdm

from cell2cell.core import interaction_space as ispace
from cell2cell.preprocessing import manipulate_dataframes, ppi, rnaseq
from cell2cell.stats.permutation import compute_pvalue_from_dist

# TODO: Implement  multiple test correction for permutation analysis (FDR, Bonferroni, etc)


class BulkInteractions:
    def __init__(self, rnaseq_data, ppi_data, metadata=None, interaction_columns=('A', 'B'),
                 communication_score='expression_thresholding', cci_score='bray_curtis', cci_type='undirected',
                 sample_col='sampleID', group_col='tissue', expression_threshold=10, complex_sep=None, verbose=False):
        # Placeholders
        self.rnaseq_data = rnaseq_data
        self.metadata = metadata
        self.index_col = sample_col
        self.group_col = group_col
        self.analysis_setup = dict()
        self.cutoff_setup = dict()
        self.complex_sep = complex_sep
        self.interaction_columns = interaction_columns

        # Analysis
        self.analysis_setup['communication_score'] = communication_score
        self.analysis_setup['cci_score'] = cci_score
        self.analysis_setup['cci_type'] = cci_type

        # Initialize PPI
        genes = list(rnaseq_data.index)
        ppi_data_ = ppi.filter_ppi_by_proteins(ppi_data=ppi_data,
                                               proteins=genes,
                                               complex_sep=complex_sep,
                                               upper_letter_comparison=False,
                                               interaction_columns=self.interaction_columns)

        self.ppi_data = ppi.remove_ppi_bidirectionality(ppi_data=ppi_data_,
                                                        interaction_columns=self.interaction_columns,
                                                        verbose=verbose)
        if self.analysis_setup['cci_type'] == 'undirected':
            self.ppi_data = ppi.bidirectional_ppi_for_cci(ppi_data=self.ppi_data,
                                                          interaction_columns=self.interaction_columns,
                                                          verbose=verbose)
            self.ref_ppi = self.ppi_data
        else:
            self.ref_ppi = None

        # Thresholding
        self.cutoff_setup['type'] = 'constant_value'
        self.cutoff_setup['parameter'] = expression_threshold

        # Interaction Space
        self.interaction_space = initialize_interaction_space(rnaseq_data=self.rnaseq_data,
                                                              ppi_data=self.ppi_data,
                                                              cutoff_setup=self.cutoff_setup,
                                                              analysis_setup=self.analysis_setup,
                                                              complex_sep=complex_sep,
                                                              interaction_columns=self.interaction_columns,
                                                              verbose=verbose)

    def compute_pairwise_cci_scores(self, cci_score=None, use_ppi_score=False, verbose=True):
        self.interaction_space.compute_pairwise_cci_scores(cci_score=cci_score,
                                                           use_ppi_score=use_ppi_score,
                                                           verbose=verbose)

    def compute_pairwise_communication_scores(self, communication_score=None, use_ppi_score=False, ref_ppi_data=None,
                                              interaction_columns=None, cells=None, cci_type=None, verbose=True):
        if interaction_columns is None:
            interaction_columns = self.interaction_columns # Used only for ref_ppi_data

        if ref_ppi_data is None:
            ref_ppi_data = self.ref_ppi

        self.interaction_space.compute_pairwise_communication_scores(communication_score=communication_score,
                                                                     use_ppi_score=use_ppi_score,
                                                                     ref_ppi_data=ref_ppi_data,
                                                                     interaction_columns=interaction_columns,
                                                                     cells=cells,
                                                                     cci_type=cci_type,
                                                                     verbose=verbose)


class SingleCellInteractions:
    compute_pairwise_cci_scores = BulkInteractions.compute_pairwise_cci_scores
    compuse_pairwise_communication_scores =  BulkInteractions.compute_pairwise_communication_scores

    def __init__(self, rnaseq_data, ppi_data, metadata=None, interaction_columns=('A', 'B'),
                 communication_score='expression_thresholding', cci_score='bray_curtis', cci_type='undirected',
                 expression_threshold=0.20, aggregation_method='nn_cell_fraction', barcode_col='barcodes',
                 celltype_col='cell_types', complex_sep=None, verbose=False):
        # Placeholders
        self.rnaseq_data = rnaseq_data
        self.metadata = metadata
        self.index_col = barcode_col
        self.group_col = celltype_col
        self.aggregation_method = aggregation_method
        self.analysis_setup = dict()
        self.cutoff_setup = dict()
        self.complex_sep = complex_sep
        self.interaction_columns = interaction_columns

        # Analysis
        self.analysis_setup['communication_score'] = communication_score
        self.analysis_setup['cci_score'] = cci_score
        self.analysis_setup['cci_type'] = cci_type

        # Initialize PPI
        genes = list(rnaseq_data.index)
        ppi_data_ = ppi.filter_ppi_by_proteins(ppi_data=ppi_data,
                                               proteins=genes,
                                               complex_sep=complex_sep,
                                               upper_letter_comparison=False,
                                               interaction_columns=interaction_columns)
        self.ppi_data = ppi.remove_ppi_bidirectionality(ppi_data=ppi_data_,
                                                        interaction_columns=interaction_columns,
                                                        verbose=verbose)
        if self.analysis_setup['cci_type'] == 'undirected':
            self.ppi_data = ppi.bidirectional_ppi_for_cci(ppi_data=self.ppi_data,
                                                          interaction_columns=interaction_columns,
                                                          verbose=verbose)
            self.ref_ppi = self.ppi_data
        else:
            self.ref_ppi = None

        # Thresholding
        self.cutoff_setup['type'] = 'constant_value'
        self.cutoff_setup['parameter'] = expression_threshold


        # Aggregate single-cell RNA-Seq data
        if isinstance(rnaseq_data, scanpy.AnnData):
            self.__adata = True
        else:
            self.__adata = False

        self.aggregated_expression = rnaseq.aggregate_single_cells(rnaseq_data=self.rnaseq_data,
                                                                   metadata=self.metadata,
                                                                   barcode_col=self.index_col,
                                                                   celltype_col=self.group_col,
                                                                   method=self.aggregation_method,
                                                                   transposed=self.__adata)

        # Interaction Space
        self.interaction_space = initialize_interaction_space(rnaseq_data=self.aggregated_expression,
                                                              ppi_data=self.ppi_data,
                                                              cutoff_setup=self.cutoff_setup,
                                                              analysis_setup=self.analysis_setup,
                                                              complex_sep=self.complex_sep,
                                                              interaction_columns=self.interaction_columns,
                                                              verbose=verbose)

    def permute_cell_labels(self, permutations=100, evaluation='communication', random_state=None, verbose=False):
        if evaluation == 'communication':
            if 'communication_matrix' not in self.interaction_space.interaction_elements.keys():
                raise ValueError('Run the method compute_pairwise_communication_scores() before permutation analysis.')
            score = self.interaction_space.interaction_elements['communication_matrix']
        elif evaluation == 'interactions':
            if not hasattr(self.interaction_space, 'distance_matrix'):
                raise ValueError('Run the method compute_pairwise_interactions() before permutation analysis.')
            score = self.interaction_space.interaction_elements['cci_matrix']
        else:
            raise ValueError('Not a valid evaluation')

        randomized_scores = []

        for i in tqdm(range(permutations), disable=not verbose):
            if random_state is not None:
                seed = random_state + i
            else:
                seed = random_state

            randomized_meta = manipulate_dataframes.shuffle_cols_in_df(df=self.metadata.reset_index(),
                                                                       columns=self.group_col,
                                                                       random_state=seed)

            aggregated_expression = rnaseq.aggregate_single_cells(rnaseq_data=self.rnaseq_data,
                                                                  metadata=randomized_meta,
                                                                  barcode_col=self.index_col,
                                                                  celltype_col=self.group_col,
                                                                  method=self.aggregation_method,
                                                                  transposed=self.__adata)

            interaction_space = initialize_interaction_space(rnaseq_data=aggregated_expression,
                                                             ppi_data=self.ppi_data,
                                                             cutoff_setup=self.cutoff_setup,
                                                             analysis_setup=self.analysis_setup,
                                                             complex_sep=self.complex_sep,
                                                             interaction_columns=self.interaction_columns,
                                                             verbose=False)

            if evaluation == 'communication':
                interaction_space.compute_pairwise_communication_scores(verbose=False)
                randomized_scores.append(interaction_space.interaction_elements['communication_matrix'].values.flatten())
            elif evaluation == 'interactions':
                interaction_space.compute_pairwise_cci_scores(verbose=False)
                randomized_scores.append(interaction_space.interaction_elements['cci_matrix'].values.flatten())

        base_scores = score.values.flatten()
        pvals = np.ones(base_scores.shape)
        for i in range(len(base_scores)):
            pvals[i] = compute_pvalue_from_dist(obs_value=base_scores[i],
                                                dist=np.array(randomized_scores)[:, i],
                                                consider_size=True,
                                                comparison='different'
                                                )
        pval_df = pd.DataFrame(pvals.reshape(score.shape), index=score.index, columns=score.columns)
        self.permutation_pvalues = pval_df
        return pval_df


class SpatialSingleCellInteractions:
    def __init__(self, rnaseq_data, ppi_data, gene_cutoffs, spatial_dict, score_type='binary', score_metric='bray_curtis',
                 n_jobs=1, verbose=True):
        pass
        # TODO IMPLEMENT SPATIAL EXPERIMENT


def initialize_interaction_space(rnaseq_data, ppi_data, cutoff_setup, analysis_setup, excluded_cells=None,
                                 complex_sep=None, interaction_columns=('A', 'B'), verbose=True):
    if excluded_cells is None:
        excluded_cells = []

    included_cells = sorted(list((set(rnaseq_data.columns) - set(excluded_cells))))

    interaction_space = ispace.InteractionSpace(rnaseq_data=rnaseq_data[included_cells],
                                                ppi_data=ppi_data,
                                                gene_cutoffs=cutoff_setup,
                                                communication_score=analysis_setup['communication_score'],
                                                cci_score=analysis_setup['cci_score'],
                                                cci_type=analysis_setup['cci_type'],
                                                complex_sep=complex_sep,
                                                interaction_columns=interaction_columns,
                                                verbose=verbose)
    return interaction_space