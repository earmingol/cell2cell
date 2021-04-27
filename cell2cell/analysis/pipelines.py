# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os

import scanpy

import cell2cell.plotting.ccc_plot
import cell2cell.plotting.cci_plot
import cell2cell.plotting.pcoa_plot
from cell2cell.core import interaction_space as ispace
from cell2cell.io import read_data
from cell2cell.preprocessing import ppi

class BulkExperiment:
    def __init__(self, rnaseq_data, ppi_data, metadata=None, interaction_columns=('A', 'B'),
                 communication_score='expression_thresholding', cci_score='bray_curtis', cci_type='undirected',
                 expression_threshold=10, verbose=False):
        # Placeholders
        self.rnaseq_data = rnaseq_data
        self.ppi_data = ppi.remove_ppi_bidirectionality(ppi_data=ppi_data,
                                                        interaction_columns=interaction_columns,
                                                        verbose=verbose)
        self.ref_ppi = self.ppi_data
        self.metadata = metadata
        self.analysis_setup = dict()
        self.cutoff_setup = dict()

        # Analysis
        self.analysis_setup['communication_score'] = communication_score
        self.analysis_setup['cci_score'] = cci_score
        self.analysis_setup['cci_type'] = cci_type

        # Initialize PPI
        if self.analysis_setup['cci_type'] == 'undirected':
            self.ppi_data = ppi.bidirectional_ppi_for_cci(ppi_data=self.ppi_data,
                                                          interaction_columns=interaction_columns,
                                                          verbose=verbose)
        else:
            self.ref_ppi = None

        # Thresholding
        self.cutoff_setup['type'] = 'constant_value'
        self.cutoff_setup['parameter'] = expression_threshold

        self.interaction_space = initialize_interaction_space(rnaseq_data=self.rnaseq_data,
                                                              ppi_data=self.ppi_data,
                                                              cutoff_setup=self.cutoff_setup,
                                                              analysis_setup=self.analysis_setup,
                                                              verbose=verbose)

    def compute_pairwise_cci_scores(self, cci_score=None, use_ppi_score=False, verbose=True):
        self.interaction_space.compute_pairwise_cci_scores(cci_score=cci_score,
                                                           use_ppi_score=use_ppi_score,
                                                           verbose=verbose)

    def compute_pairwise_communication_scores(self, communication_score=None, use_ppi_score=False, ref_ppi_data=None,
                                              interaction_columns=('A', 'B'), cells=None, cci_type=None, verbose=True):
        self.interaction_space.compute_pairwise_communication_scores(communication_score=communication_score,
                                                                     use_ppi_score=use_ppi_score,
                                                                     ref_ppi_data=ref_ppi_data,
                                                                     interaction_columns=interaction_columns,
                                                                     cells=cells,
                                                                     cci_type=cci_type,
                                                                     verbose=verbose)


class SingleCellExperiment:
    compute_pairwise_ccc_scores = BulkExperiment.compute_pairwise_cci_scores
    compuse_pairwise_communication_scores =  BulkExperiment.compute_pairwise_communication_scores


    def __init__(self, rnaseq_data, ppi_data, metadata=None, interaction_columns=('A', 'B'),
                 communication_score='expression_thresholding', cci_score='bray_curtis', cci_type='undirected',
                 expression_threshold=0.20, aggregation_method='nn_cell_fraction', barcode_col='barcodes',
                 celltype_col='cell_types', verbose=False):
        # Placeholders
        self.rnaseq_data = rnaseq_data
        self.ppi_data = ppi.remove_ppi_bidirectionality(ppi_data=ppi_data,
                                                        interaction_columns=interaction_columns,
                                                        verbose=verbose)
        self.ref_ppi = self.ppi_data
        self.metadata = metadata
        self.analysis_setup = dict()
        self.cutoff_setup = dict()

        # Analysis
        self.analysis_setup['communication_score'] = communication_score
        self.analysis_setup['cci_score'] = cci_score
        self.analysis_setup['cci_type'] = cci_type

        # Initialize PPI
        if self.analysis_setup['cci_type'] == 'undirected':
            self.ppi_data = ppi.bidirectional_ppi_for_cci(ppi_data=self.ppi_data,
                                                          interaction_columns=interaction_columns,
                                                          verbose=verbose)
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

        self.aggregated_expression = cell2cell.preprocessing.aggregate_single_cells(rnaseq_data=self.rnaseq_data,
                                                                                    metadata=self.metadata,
                                                                                    barcode_col=barcode_col,
                                                                                    celltype_col=celltype_col,
                                                                                    method=aggregation_method,
                                                                                    transposed=self.__adata)

        self.interaction_space = initialize_interaction_space(rnaseq_data=self.aggregated_expression,
                                                              ppi_data=self.ppi_data,
                                                              cutoff_setup=self.cutoff_setup,
                                                              analysis_setup=self.analysis_setup,
                                                              verbose=verbose)


class SpatialSingleCellExperiment:
    def __init__(self, rnaseq_data, ppi_data, gene_cutoffs, spatial_dict, score_type='binary', score_metric='bray_curtis',
                 n_jobs=1, verbose=True):
        pass
        # TODO IMPLEMENT SPATIAL EXPERIMENT


def initialize_interaction_space(rnaseq_data, ppi_data, cutoff_setup, analysis_setup, excluded_cells=None, verbose=True):
    if excluded_cells is None:
        excluded_cells = []

    included_cells = sorted(list((set(rnaseq_data.columns) - set(excluded_cells))))

    interaction_space = ispace.InteractionSpace(rnaseq_data=rnaseq_data[included_cells],
                                                ppi_data=ppi_data,
                                                gene_cutoffs=cutoff_setup,
                                                communication_score=analysis_setup['communication_score'],
                                                cci_score=analysis_setup['cci_score'],
                                                cci_type=analysis_setup['cci_type'],
                                                verbose=verbose)
    return interaction_space


def core_pipeline(files, rnaseq_data, ppi_data, metadata, meta_setup, cutoff_setup, analysis_setup, excluded_cells=None,
                  use_ppi_score=False, make_plots=True, colors=None, ccc_clustermap_metric='jaccard', filename_suffix='',
                  verbose=True):

    if excluded_cells is None:
        excluded_cells = []

    # Generate reference ppi data for generating directed CCC clustermap later
    if analysis_setup['cci_type'] == 'undirected':
        bi_ppi_data = ppi.bidirectional_ppi_for_cci(ppi_data=ppi_data, verbose=False)
        ref_ppi = ppi_data
    else:
        bi_ppi_data = ppi_data.copy()
        ref_ppi = None

    # Generate interaction space and compute parwise CCI scores
    interaction_space = initialize_interaction_space(rnaseq_data=rnaseq_data,
                                                     ppi_data=bi_ppi_data,
                                                     cutoff_setup=cutoff_setup,
                                                     analysis_setup=analysis_setup,
                                                     excluded_cells=excluded_cells,
                                                     verbose=verbose)

    # Compute CCI scores for each pair of cell
    interaction_space.compute_pairwise_cci_scores(use_ppi_score=use_ppi_score,
                                                  verbose=verbose)

    # Compute communication scores for each protein-protein interaction
    interaction_space.compute_pairwise_communication_scores(ref_ppi_data=ref_ppi,
                                                            use_ppi_score=use_ppi_score,
                                                            verbose=verbose)

    # Plots section
    if make_plots:
        clustermap = cell2cell.plotting.cci_plot.clustermap_cci(interaction_space,
                                                                method='ward',
                                                                excluded_cells=excluded_cells,
                                                                metadata=metadata,
                                                                sample_col=meta_setup['sample_col'],
                                                                group_col=meta_setup['group_col'],
                                                                colors=colors,
                                                                title='CCI scores for cell pairs',
                                                                filename=files['output_folder'] + 'CCI-Clustermap-CCI-scores{}.png'.format(filename_suffix),
                                                                **{'cmap': 'Blues'}
                                                                )

        # Run PCoA only if CCI matrix is symmetric
        pcoa_state = False
        if (interaction_space.interaction_elements['cci_matrix'].values.transpose() == interaction_space.interaction_elements['cci_matrix'].values).all():
            pcoa = cell2cell.plotting.pcoa_plot.pcoa_3dplot(interaction_space,
                                                            excluded_cells=excluded_cells,
                                                            metadata=metadata,
                                                            sample_col=meta_setup['sample_col'],
                                                            group_col=meta_setup['group_col'],
                                                            colors=colors,
                                                            title='PCoA for cells given their CCI scores',
                                                            filename=files['output_folder'] + 'CCI-PCoA-CCI-scores{}.png'.format(filename_suffix),
                                                            )
            pcoa_state = True

        interaction_clustermap = cell2cell.plotting.ccc_plot.clustermap_ccc(interaction_space,
                                                                            metric=ccc_clustermap_metric,
                                                                            metadata=metadata,
                                                                            sample_col=meta_setup['sample_col'],
                                                                            group_col=meta_setup['group_col'],
                                                                            colors=colors,
                                                                            excluded_cells=excluded_cells,
                                                                            title='Active ligand-receptor pairs for interacting cells',
                                                                            filename=files['output_folder'] + 'CCI-Active-LR-pairs{}.png'.format(filename_suffix),
                                                                            **{'figsize': (20, 40)}
                                                                            )

        interaction_clustermap.data2d.to_csv(files['output_folder'] + 'CCI-Active-LR-pairs{}.csv'.format(filename_suffix))

    # Save results
    outputs = dict()
    outputs['interaction_space'] = interaction_space
    if make_plots:
        outputs['cci_clustermap'] = clustermap
        if pcoa_state:
            outputs['pcoa'] = pcoa
        else:
            outputs['pcoa'] = None
        outputs['ccc_clustermap'] = interaction_clustermap
        outputs['LR-pairs'] = interaction_clustermap.data2d
    else:
        outputs['cci_clustermap'] = None
        outputs['pcoa'] = None
        outputs['ccc_clustermap'] = None
        outputs['LR-pairs'] = None
    return outputs


def ligand_receptor_pipeline(files, rnaseq_setup, ppi_setup, meta_setup, cutoff_setup, analysis_setup, excluded_cells=None,
                             colors=None,  use_ppi_score=False, filename_suffix='', verbose=True):
    if excluded_cells is None:
        excluded_cells = []

    # Check for output directory
    if 'output_folder' in files.keys():
        if not os.path.exists(files['output_folder']):
            os.makedirs(files['output_folder'])

    # Load Data
    rnaseq_data = read_data.load_rnaseq(rnaseq_file=files['rnaseq'],
                                        gene_column=rnaseq_setup['gene_col'],
                                        drop_nangenes=rnaseq_setup['drop_nangenes'],
                                        log_transformation=rnaseq_setup['log_transform'],
                                        format='auto',
                                        verbose=verbose)

    ppi_data = read_data.load_ppi(ppi_file=files['ppi'],
                                  interaction_columns=ppi_setup['protein_cols'],
                                  rnaseq_genes=list(rnaseq_data.index),
                                  format='auto',
                                  verbose=verbose)

    meta = read_data.load_metadata(metadata_file=files['metadata'],
                                   rnaseq_data=rnaseq_data,
                                   sample_col=meta_setup['sample_col'],
                                   format='auto',
                                   verbose=verbose)

    # Run Analysis
    outputs = core_pipeline(files=files,
                            rnaseq_data=rnaseq_data,
                            ppi_data=ppi_data,
                            metadata=meta,
                            meta_setup=meta_setup,
                            cutoff_setup=cutoff_setup,
                            analysis_setup=analysis_setup,
                            excluded_cells=excluded_cells,
                            colors=colors,
                            use_ppi_score=use_ppi_score,
                            filename_suffix=filename_suffix,
                            verbose=verbose)
    return outputs