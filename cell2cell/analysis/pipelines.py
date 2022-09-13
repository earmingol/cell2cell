# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd
import scanpy
import numpy as np
from tqdm import tqdm

from cell2cell.core import interaction_space as ispace
from cell2cell.preprocessing import manipulate_dataframes, ppi, rnaseq
from cell2cell.stats import permutation, multitest


class BulkInteractions:
    '''Interaction class with all necessary methods to run the cell2cell pipeline
    on a bulk RNA-seq dataset. Cells here could be represented by tissues, samples
    or any bulk organization of cells.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment. Columns are samples
        and rows are genes.

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    metadata : pandas.Dataframe, default=None
        Metadata associated with the samples in the RNA-seq dataset.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    communication_score : str, default='expression_thresholding'
        Type of communication score to infer the potential use of a given ligand-
        receptor pair by a pair of cells/tissues/samples.
        Available communication_scores are:

        - 'expression_thresholding' : Computes the joint presence of a ligand from a
                                     sender cell and of a receptor on a receiver cell
                                     from binarizing their gene expression levels.
        - 'expression_mean' : Computes the average between the expression of a ligand
                              from a sender cell and the expression of a receptor on a
                              receiver cell.
        - 'expression_product' : Computes the product between the expression of a
                                ligand from a sender cell and the expression of a
                                receptor on a receiver cell.
        - 'expression_gmean' : Computes the geometric mean between the expression
                              of a ligand from a sender cell and the
                              expression of a receptor on a receiver cell.

    cci_score : str, default='bray_curtis'
        Scoring function to aggregate the communication scores between a pair of
        cells. It computes an overall potential of cell-cell interactions.
        Options:

        - 'bray_curtis' : Bray-Curtis-like score.
        - 'jaccard' : Jaccard-like score.
        - 'count' : Number of LR pairs that the pair of cells use.
        - 'icellnet' : Sum of the L-R expression product of a pair of cells

    cci_type : str, default='undirected'
        Specifies whether computing the cci_score in a directed or undirected
        way. For a pair of cells A and B, directed means that the ligands are
        considered only from cell A and receptors only from cell B or viceversa.
        While undirected simultaneously considers signaling from cell A to
        cell B and from cell B to cell A.

    sample_col : str, default='sampleID'
        Column-name for the samples in the metadata.

    group_col : str, default='tissue'
        Column-name for the grouping information associated with the samples
        in the metadata.

    expression_threshold : float, default=10
        Threshold value to binarize gene expression when using
        communication_score='expression_thresholding'. Units have to be the
        same as the rnaseq_data matrix (e.g., TPMs, counts, etc.).

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    complex_agg_method : str, default='min'
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.
        - 'gmean' : Geometric mean expression value among all genes.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    Attributes
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment. Columns are samples
        and rows are genes.

    metadata : pandas.DataFrame
        Metadata associated with the samples in the RNA-seq dataset.

    index_col : str
        Column-name for the samples in the metadata.

    group_col : str
        Column-name for the grouping information associated with the samples
        in the metadata.

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used for
        inferring the cell-cell interactions and communication.

    complex_sep : str
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    complex_agg_method : str
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.
        - 'gmean' : Geometric mean expression value among all genes.

    ref_ppi : pandas.DataFrame
        Reference list of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication. It could be the
        same as 'ppi_data' if ppi_data is not bidirectional (that is, contains
        ProtA-ProtB interaction as well as ProtB-ProtA interaction). ref_ppi must
        be undirected (contains only ProtA-ProtB and not ProtB-ProtA interaction).

    interaction_columns : tuple
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    analysis_setup : dict
        Contains main setup for running the cell-cell interactions and communication
        analyses.
        Three main setups are needed (passed as keys):

        - 'communication_score' : is the type of communication score used to detect
            active ligand-receptor pairs between each pair of cell.
            It can be:

            - 'expression_thresholding'
            - 'expression_product'
            - 'expression_mean'
            - 'expression_gmean'

        - 'cci_score' : is the scoring function to aggregate the communication
            scores.
            It can be:

            - 'bray_curtis'
            - 'jaccard'
            - 'count'
            - 'icellnet'

        - 'cci_type' : is the type of interaction between two cells. If it is
            undirected, all ligands and receptors are considered from both cells.
            If it is directed, ligands from one cell and receptors from the other
            are considered separately with respect to ligands from the second
            cell and receptor from the first one.
            So, it can be:

            - 'undirected'
            - 'directed'

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

    interaction_space : cell2cell.core.interaction_space.InteractionSpace
        Interaction space that contains all the required elements to perform the
        cell-cell interaction and communication analysis between every pair of cells.
        After performing the analyses, the results are stored in this object.
    '''
    def __init__(self, rnaseq_data, ppi_data, metadata=None, interaction_columns=('A', 'B'),
                 communication_score='expression_thresholding', cci_score='bray_curtis', cci_type='undirected',
                 sample_col='sampleID', group_col='tissue', expression_threshold=10, complex_sep=None,
                 complex_agg_method='min', verbose=False):
        # Placeholders
        self.rnaseq_data = rnaseq_data
        self.metadata = metadata
        self.index_col = sample_col
        self.group_col = group_col
        self.analysis_setup = dict()
        self.cutoff_setup = dict()
        self.complex_sep = complex_sep
        self.complex_agg_method = complex_agg_method
        self.interaction_columns = interaction_columns

        # Analysis setup
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
            self.ref_ppi = self.ppi_data.copy()
            self.ppi_data = ppi.bidirectional_ppi_for_cci(ppi_data=self.ppi_data,
                                                          interaction_columns=self.interaction_columns,
                                                          verbose=verbose)
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
                                                              complex_agg_method=complex_agg_method,
                                                              interaction_columns=self.interaction_columns,
                                                              verbose=verbose)

    def compute_pairwise_cci_scores(self, cci_score=None, use_ppi_score=False, verbose=True):
        '''Computes overall CCI scores for each pair of cells.

        Parameters
        ----------
        cci_score : str, default=None
            Scoring function to aggregate the communication scores between
            a pair of cells. It computes an overall potential of cell-cell
            interactions. If None, it will use the one stored in the
            attribute analysis_setup of this object.
            Options:

            - 'bray_curtis' : Bray-Curtis-like score.
            - 'jaccard' : Jaccard-like score.
            - 'count' : Number of LR pairs that the pair of cells use.
            - 'icellnet' : Sum of the L-R expression product of a pair of cells

        use_ppi_score : boolean, default=False
            Whether using a weight of LR pairs specified in the ppi_data
            to compute the scores.

        verbose : boolean, default=True
            Whether printing or not steps of the analysis.
        '''
        self.interaction_space.compute_pairwise_cci_scores(cci_score=cci_score,
                                                           use_ppi_score=use_ppi_score,
                                                           verbose=verbose)

    def compute_pairwise_communication_scores(self, communication_score=None, use_ppi_score=False, ref_ppi_data=None,
                                              interaction_columns=None, cells=None, cci_type=None, verbose=True):
        '''Computes the communication scores for each LR pairs in
        a given pair of sender-receiver cell

        Parameters
        ----------
        communication_score : str, default=None
            Type of communication score to infer the potential use of
            a given ligand-receptor pair by a pair of cells/tissues/samples.
            If None, the score stored in the attribute analysis_setup
            will be used.
            Available communication_scores are:

            - 'expresion_thresholding' : Computes the joint presence of a
                                         ligand from a sender cell and of
                                         a receptor on a receiver cell from
                                         binarizing their gene expression levels.
            - 'expression_mean' : Computes the average between the expression
                                  of a ligand from a sender cell and the
                                  expression of a receptor on a receiver cell.
            - 'expression_product' : Computes the product between the expression
                                    of a ligand from a sender cell and the
                                    expression of a receptor on a receiver cell.
            - 'expression_gmean' : Computes the geometric mean between the expression
                                  of a ligand from a sender cell and the
                                  expression of a receptor on a receiver cell.

        use_ppi_score : boolean, default=False
            Whether using a weight of LR pairs specified in the ppi_data
            to compute the scores.

        ref_ppi_data : pandas.DataFrame, default=None
            Reference list of protein-protein interactions (or
            ligand-receptor pairs) used for inferring the cell-cell
            interactions and communication. It could be the same as
            'ppi_data' if ppi_data is not bidirectional (that is,
            contains ProtA-ProtB interaction as well as ProtB-ProtA
            interaction). ref_ppi must be undirected (contains only
            ProtA-ProtB and not ProtB-ProtA interaction). If None
            the one stored in the attribute ref_ppi will be used.

        interaction_columns : tuple, default=None
            Contains the names of the columns where to find the
            partners in a dataframe of protein-protein interactions.
            If the list is for ligand-receptor pairs, the first column
            is for the ligands and the second for the receptors. If
            None, the one stored in the attribute interaction_columns
            will be used.

        cells : list=None
            List of cells to consider.

        cci_type : str, default=None
            Type of interaction between two cells. Used to specify
            if we want to consider a LR pair in both directions.
            It can be:

            - 'undirected'
            - 'directed'

            If None, the one stored in the attribute analysis_setup
            will be used.

        verbose : boolean, default=True
            Whether printing or not steps of the analysis.
        '''
        if interaction_columns is None:
            interaction_columns = self.interaction_columns # Used only for ref_ppi_data

        if ref_ppi_data is None:
            ref_ppi_data = self.ref_ppi

        if cci_type is None:
            cci_type = 'directed'

        self.interaction_space.compute_pairwise_communication_scores(communication_score=communication_score,
                                                                     use_ppi_score=use_ppi_score,
                                                                     ref_ppi_data=ref_ppi_data,
                                                                     interaction_columns=interaction_columns,
                                                                     cells=cells,
                                                                     cci_type=cci_type,
                                                                     verbose=verbose)


class SingleCellInteractions:
    '''Interaction class with all necessary methods to run the cell2cell pipeline
        on a single-cell RNA-seq dataset.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame or scanpy.AnnData
        Gene expression data for a single-cell RNA-seq experiment. If it is a
        dataframe columns are single cells and rows are genes, while if it is
        a AnnData object, columns are genes and rows are single cells.

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    metadata : pandas.Dataframe
        Metadata containing the cell types for each single cells in the
        RNA-seq dataset.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    communication_score : str, default='expression_thresholding'
        Type of communication score to infer the potential use of a given ligand-
        receptor pair by a pair of cells/tissues/samples.
        Available communication_scores are:

        - 'expression_thresholding' : Computes the joint presence of a ligand from a
                                     sender cell and of a receptor on a receiver cell
                                     from binarizing their gene expression levels.
        - 'expression_mean' : Computes the average between the expression of a ligand
                              from a sender cell and the expression of a receptor on a
                              receiver cell.
        - 'expression_product' : Computes the product between the expression of a
                                ligand from a sender cell and the expression of a
                                receptor on a receiver cell.
        - 'expression_gmean' : Computes the geometric mean between the expression
                              of a ligand from a sender cell and the
                              expression of a receptor on a receiver cell.

    cci_score : str, default='bray_curtis'
        Scoring function to aggregate the communication scores between a pair of
        cells. It computes an overall potential of cell-cell interactions.
        Options:

        - 'bray_curtis' : Bray-Curtis-like score.
        - 'jaccard' : Jaccard-like score.
        - 'count' : Number of LR pairs that the pair of cells use.
        - 'icellnet' : Sum of the L-R expression product of a pair of cells

    cci_type : str, default='undirected'
        Specifies whether computing the cci_score in a directed or undirected
        way. For a pair of cells A and B, directed means that the ligands are
        considered only from cell A and receptors only from cell B or viceversa.
        While undirected simultaneously considers signaling from cell A to
        cell B and from cell B to cell A.

    expression_threshold : float, default=0.2
        Threshold value to binarize gene expression when using
        communication_score='expression_thresholding'. Units have to be the
        same as the aggregated gene expression matrix (e.g., counts, fraction
        of cells with non-zero counts, etc.).

    aggregation_method : str, default='nn_cell_fraction'
        Specifies the method to use to aggregate gene expression of single
        cells into their respective cell types. Used to perform the CCI
        analysis since it is on the cell types rather than single cells.
        Options are:

        - 'nn_cell_fraction' : Among the single cells composing a cell type, it
            calculates the fraction of single cells with non-zero count values
            of a given gene.
        - 'average' : Computes the average gene expression among the single cells
            composing a cell type for a given gene.

    barcode_col : str, default='barcodes'
        Column-name for the single cells in the metadata.

    celltype_col : str, default='celltypes'
        Column-name in the metadata for the grouping single cells into cell types
        by the selected aggregation method.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    complex_agg_method : str, default='min'
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.
        - 'gmean' : Geometric mean expression value among all genes.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    Attributes
    ----------
    rnaseq_data : pandas.DataFrame or scanpy.AnnData
        Gene expression data for a single-cell RNA-seq experiment. If it is a
        dataframe columns are single cells and rows are genes, while if it is
        a AnnData object, columns are genes and rows are single cells.

    metadata : pandas.DataFrame
        Metadata containing the cell types for each single cells in the
        RNA-seq dataset.

    index_col : str
        Column-name for the single cells in the metadata.

    group_col : str
        Column-name in the metadata for the grouping single cells into cell types
        by the selected aggregation method.

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used for
        inferring the cell-cell interactions and communication.

    complex_sep : str
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    complex_agg_method : str
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.
        - 'gmean' : Geometric mean expression value among all genes.

    ref_ppi : pandas.DataFrame
        Reference list of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication. It could be the
        same as 'ppi_data' if ppi_data is not bidirectional (that is, contains
        ProtA-ProtB interaction as well as ProtB-ProtA interaction). ref_ppi must
        be undirected (contains only ProtA-ProtB and not ProtB-ProtA interaction).

    interaction_columns : tuple
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    analysis_setup : dict
        Contains main setup for running the cell-cell interactions and communication
        analyses.
        Three main setups are needed (passed as keys):

        - 'communication_score' : is the type of communication score used to detect
            active ligand-receptor pairs between each pair of cell.
            It can be:

            - 'expression_thresholding'
            - 'expression_product'
            - 'expression_mean'
            - 'expression_gmean'

        - 'cci_score' : is the scoring function to aggregate the communication
            scores.
            It can be:

            - 'bray_curtis'
            - 'jaccard'
            - 'count'
            - 'icellnet'

        - 'cci_type' : is the type of interaction between two cells. If it is
            undirected, all ligands and receptors are considered from both cells.
            If it is directed, ligands from one cell and receptors from the other
            are considered separately with respect to ligands from the second
            cell and receptor from the first one.
            So, it can be:

            - 'undirected'
            - 'directed'

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

    interaction_space : cell2cell.core.interaction_space.InteractionSpace
        Interaction space that contains all the required elements to perform the
        cell-cell interaction and communication analysis between every pair of cells.
        After performing the analyses, the results are stored in this object.

    aggregation_method : str
        Specifies the method to use to aggregate gene expression of single
        cells into their respective cell types. Used to perform the CCI
        analysis since it is on the cell types rather than single cells.
        Options are:

        - 'nn_cell_fraction' : Among the single cells composing a cell type, it
            calculates the fraction of single cells with non-zero count values
            of a given gene.
        - 'average' : Computes the average gene expression among the single cells
            composing a cell type for a given gene.

    ccc_permutation_pvalues : pandas.DataFrame
        Contains the P-values of the permutation analysis on the
        communication scores.

    cci_permutation_pvalues : pandas.DataFrame
        Contains the P-values of the permutation analysis on the
        CCI scores.

    __adata : boolean
        Auxiliary variable used for storing whether rnaseq_data
        is an AnnData object.
    '''
    compute_pairwise_cci_scores = BulkInteractions.compute_pairwise_cci_scores
    compute_pairwise_communication_scores =  BulkInteractions.compute_pairwise_communication_scores

    def __init__(self, rnaseq_data, ppi_data, metadata, interaction_columns=('A', 'B'),
                 communication_score='expression_thresholding', cci_score='bray_curtis', cci_type='undirected',
                 expression_threshold=0.20, aggregation_method='nn_cell_fraction', barcode_col='barcodes',
                 celltype_col='cell_types', complex_sep=None, complex_agg_method='min', verbose=False):
        # Placeholders
        self.rnaseq_data = rnaseq_data
        self.metadata = metadata
        self.index_col = barcode_col
        self.group_col = celltype_col
        self.aggregation_method = aggregation_method
        self.analysis_setup = dict()
        self.cutoff_setup = dict()
        self.complex_sep = complex_sep
        self.complex_agg_method = complex_agg_method
        self.interaction_columns = interaction_columns
        self.ccc_permutation_pvalues = None
        self.cci_permutation_pvalues = None

        if isinstance(rnaseq_data, scanpy.AnnData):
            self.__adata = True
            genes = list(rnaseq_data.var.index)
        else:
            self.__adata = False
            genes = list(rnaseq_data.index)

        # Analysis
        self.analysis_setup['communication_score'] = communication_score
        self.analysis_setup['cci_score'] = cci_score
        self.analysis_setup['cci_type'] = cci_type

        # Initialize PPI
        ppi_data_ = ppi.filter_ppi_by_proteins(ppi_data=ppi_data,
                                               proteins=genes,
                                               complex_sep=complex_sep,
                                               upper_letter_comparison=False,
                                               interaction_columns=interaction_columns)

        self.ppi_data = ppi.remove_ppi_bidirectionality(ppi_data=ppi_data_,
                                                        interaction_columns=interaction_columns,
                                                        verbose=verbose)

        if self.analysis_setup['cci_type'] == 'undirected':
            self.ref_ppi = self.ppi_data
            self.ppi_data = ppi.bidirectional_ppi_for_cci(ppi_data=self.ppi_data,
                                                          interaction_columns=interaction_columns,
                                                          verbose=verbose)
        else:
            self.ref_ppi = None

        # Thresholding
        self.cutoff_setup['type'] = 'constant_value'
        self.cutoff_setup['parameter'] = expression_threshold


        # Aggregate single-cell RNA-Seq data
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
                                                              complex_agg_method=self.complex_agg_method,
                                                              interaction_columns=self.interaction_columns,
                                                              verbose=verbose)

    def permute_cell_labels(self, permutations=100, evaluation='communication', fdr_correction=True, random_state=None,
                            verbose=False):
        '''Performs permutation analysis of cell-type labels. Detects
        significant CCI or communication scores.

        Parameters
        ----------
        permutations : int, default=100
            Number of permutations where in each of them a random
            shuffle of cell-type labels is performed, followed of
            computing CCI or communication scores to create a null
            distribution.

        evaluation : str, default='communication'
            Whether calculating P-values for CCI or communication scores.

            - 'interactions' : For CCI scores.
            - 'communication' : For communication scores.

        fdr_correction : boolean, default=True
            Whether performing a multiple test correction after
            computing P-values. In this case corresponds to an
            FDR or Benjamini-Hochberg correction, using an alpha
            equal to 0.05.

        random_state : int, default=None
            Seed for randomization.

        verbose : boolean, default=False
            Whether printing or not steps of the analysis.
        '''
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
                                                             complex_agg_method=self.complex_agg_method,
                                                             interaction_columns=self.interaction_columns,
                                                             verbose=False)

            if evaluation == 'communication':
                interaction_space.compute_pairwise_communication_scores(verbose=False)
                randomized_scores.append(interaction_space.interaction_elements['communication_matrix'].values.flatten())
            elif evaluation == 'interactions':
                interaction_space.compute_pairwise_cci_scores(verbose=False)
                randomized_scores.append(interaction_space.interaction_elements['cci_matrix'].values.flatten())

        randomized_scores = np.array(randomized_scores)
        base_scores = score.values.flatten()
        pvals = np.ones(base_scores.shape)
        for i in range(len(base_scores)):
            dist = randomized_scores[:, i]
            dist = np.append(dist, base_scores[i])
            pvals[i] = permutation.compute_pvalue_from_dist(obs_value=base_scores[i],
                                                            dist=dist,
                                                            consider_size=True,
                                                            comparison='different'
                                                            )
        pval_df = pd.DataFrame(pvals.reshape(score.shape), index=score.index, columns=score.columns)

        if fdr_correction:
            symmetric = manipulate_dataframes.check_symmetry(df=pval_df)
            if symmetric:
                pval_df = multitest.compute_fdrcorrection_symmetric_matrix(X=pval_df,
                                                                           alpha=0.05)
            else:
                pval_df = multitest.compute_fdrcorrection_asymmetric_matrix(X=pval_df,
                                                                            alpha=0.05)

        if evaluation == 'communication':
            self.ccc_permutation_pvalues = pval_df
        elif evaluation == 'interactions':
            self.cci_permutation_pvalues = pval_df
        return pval_df


class SpatialSingleCellInteractions:
    def __init__(self, rnaseq_data, ppi_data, gene_cutoffs, spatial_dict, score_type='binary', score_metric='bray_curtis',
                 n_jobs=1, verbose=True):
        pass
        # TODO IMPLEMENT SPATIAL EXPERIMENT


def initialize_interaction_space(rnaseq_data, ppi_data, cutoff_setup, analysis_setup, excluded_cells=None,
                                 complex_sep=None, complex_agg_method='min', interaction_columns=('A', 'B'),
                                 verbose=True):
    '''Initializes a InteractionSpace object to perform the analyses

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment or a single-cell
        experiment after aggregation into cell types. Columns are samples
        and rows are genes.

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

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

    analysis_setup : dict
        Contains main setup for running the cell-cell interactions and communication
        analyses.
        Three main setups are needed (passed as keys):

        - 'communication_score' : is the type of communication score used to detect
            active ligand-receptor pairs between each pair of cell.
            It can be:

            - 'expression_thresholding'
            - 'expression_product'
            - 'expression_mean'
            - 'expression_gmean'

        - 'cci_score' : is the scoring function to aggregate the communication
            scores.
            It can be:

            - 'bray_curtis'
            - 'jaccard'
            - 'count'
            - 'icellnet'

        - 'cci_type' : is the type of interaction between two cells. If it is
            undirected, all ligands and receptors are considered from both cells.
            If it is directed, ligands from one cell and receptors from the other
            are considered separately with respect to ligands from the second
            cell and receptor from the first one.
            So, it can be:

            - 'undirected'
            - 'directed'

    excluded_cells : list, default=None
        List of cells in the rnaseq_data to be excluded. If None, all cells
        are considered.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    complex_agg_method : str, default='min'
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    interaction_space : cell2cell.core.interaction_space.InteractionSpace
        Interaction space that contains all the required elements to perform the
        cell-cell interaction and communication analysis between every pair of cells.
        After performing the analyses, the results are stored in this object.
    '''
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
                                                complex_agg_method=complex_agg_method,
                                                interaction_columns=interaction_columns,
                                                verbose=verbose)
    return interaction_space