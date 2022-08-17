# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.preprocessing import integrate_data, cutoffs, get_genes_from_complexes, add_complexes_to_expression
from cell2cell.core import cell, cci_scores, communication_scores

import itertools
import numpy as np
import pandas as pd


def generate_pairs(cells, cci_type, self_interaction=True, remove_duplicates=True):
    '''Generates a list of pairs of interacting cell-types/tissues/samples.

    Parameters
    ----------
    cells : list
        A lyst of cell-type/tissue/sample names.

    cci_type : str,
        Type of interactions.
        Options are:

        - 'directed' : Directed cell-cell interactions, so pair A-B is different
            to pair B-A and both are considered.
        - 'undirected' : Undirected cell-cell interactions, so pair A-B is equal
            to pair B-A and just one of them is considered.

    self_interaction : boolean, default=True
        Whether considering autocrine interactions (pair A-A, B-B, etc).

    remove_duplicates : boolean, default=True
        Whether removing duplicates when a list of cells is passed and names are
        duplicated. If False and a list [A, A, B] is passed, pairs could be
        [A-A, A-A, A-B, A-A, A-A, A-B, B-A, B-A, B-B] when self_interaction is True
        and cci_type is 'directed'. In the same scenario but when remove_duplicates
        is True, the resulting list would be [A-A, A-B, B-A, B-B].

    Returns
    -------
    pairs : list
        List with pairs of interacting cell-types/tissues/samples.
    '''
    if self_interaction:
        if cci_type == 'directed':
            pairs = list(itertools.product(cells, cells))
            #pairs = list(itertools.combinations(cells + cells, 2)) # Directed
        elif cci_type == 'undirected':
            pairs = list(itertools.combinations(cells, 2)) + [(c, c) for c in cells] # Undirected
        else:
            raise NotImplementedError("CCI type has to be directed or undirected")
    else:
        if cci_type == 'directed':
            pairs_ = list(itertools.product(cells, cells))
            pairs = []
            for p in pairs_:
                if p[0] == p[1]:
                    continue
                else:
                    pairs.append(p)
        elif cci_type == 'undirected':
            pairs = list(itertools.combinations(cells, 2))
        else:
            raise NotImplementedError("CCI type has to be directed or undirected")
    if remove_duplicates:
        pairs = list(set(pairs))  # Remove duplicates
    return pairs


def generate_interaction_elements(modified_rnaseq, ppi_data, cci_type='undirected', cci_matrix_template=None,
                                  complex_sep=None, complex_agg_method='min', interaction_columns=('A', 'B'),
                                  verbose=True):
    '''Create all elements needed to perform the analyses of pairwise
    cell-cell interactions/communication. Corresponds to the interaction
    elements used by the class InteractionSpace.

    Parameters
    ----------
    modified_rnaseq : pandas.DataFrame
        Preprocessed gene expression data for a bulk or single-cell RNA-seq experiment.
        Columns are are cell-types/tissues/samples and rows are genes. The preprocessing
        may correspond to scoring the gene expression as binary or continuous values
        depending on the scoring function for cell-cell interactions/communication.

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used for
        inferring the cell-cell interactions and communication.

    cci_type : str, default='undirected'
        Specifies whether computing the cci_score in a directed or undirected
        way. For a pair of cells A and B, directed means that the ligands are
        considered only from cell A and receptors only from cell B or viceversa.
        While undirected simultaneously considers signaling from cell A to
        cell B and from cell B to cell A.

    cci_matrix_template : pandas.DataFrame, default=None
        A matrix of shape MxM where M are cell-types/tissues/samples. This
        is used as template for storing CCI scores. It may be useful
        for specifying which pairs of cells to consider.

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

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    interaction_elements : dict
        Dictionary containing all the pairs of cells considered (under
        the key of 'pairs'), Cell instances (under key 'cells')
        which include all cells/tissues/organs with their associated datasets
        (rna_seq, weighted_ppi, etc) and a Cell-Cell Interaction Matrix
        to store CCI scores(under key 'cci_matrix'). A communication matrix
        is also stored in this object when the communication scores are
        computed in the InteractionSpace class (under key
        'communication_score')
    '''

    if verbose:
        print('Creating Interaction Space')

    # Include complex expression
    if complex_sep is not None:
        col_a_genes, complex_a, col_b_genes, complex_b, complexes = get_genes_from_complexes(ppi_data=ppi_data,
                                                                                             complex_sep=complex_sep,
                                                                                             interaction_columns=interaction_columns
                                                                                             )
        modified_rnaseq = add_complexes_to_expression(rnaseq_data=modified_rnaseq,
                                                      complexes=complexes,
                                                      agg_method=complex_agg_method
                                                      )

    # Cells
    cell_instances = list(modified_rnaseq.columns)  # @Erick, check if position 0 of columns contain index header.
    cell_number = len(cell_instances)

    # Generate pairwise interactions
    pairwise_interactions = generate_pairs(cell_instances, cci_type)

    # Interaction elements
    interaction_elements = {}
    interaction_elements['cell_names'] = cell_instances
    interaction_elements['pairs'] = pairwise_interactions
    interaction_elements['cells'] = cell.get_cells_from_rnaseq(modified_rnaseq, verbose=verbose)

    # Cell-specific scores in PPIs

    # For matmul functions
    #interaction_elements['A_score'] = np.array([], dtype=np.int64)#.reshape(ppi_data.shape[0],0)
    #interaction_elements['B_score'] = np.array([], dtype=np.int64)#.reshape(ppi_data.shape[0],0)

    # For 'for' loop
    for cell_instance in interaction_elements['cells'].values():
        cell_instance.weighted_ppi = integrate_data.get_weighted_ppi(ppi_data=ppi_data,
                                                                     modified_rnaseq_data=cell_instance.rnaseq_data,
                                                                     column='value', # value is in each cell
                                                                     )
        #interaction_elements['A_score'] = np.hstack([interaction_elements['A_score'], cell_instance.weighted_ppi['A'].values])
        #interaction_elements['B_score'] = np.hstack([interaction_elements['B_score'], cell_instance.weighted_ppi['B'].values])

    # Cell-cell interaction matrix
    if cci_matrix_template is None:
        interaction_elements['cci_matrix'] = pd.DataFrame(np.zeros((cell_number, cell_number)),
                                                          columns=cell_instances,
                                                          index=cell_instances)
    else:
        interaction_elements['cci_matrix'] = cci_matrix_template
    return interaction_elements


class InteractionSpace():
    '''
    Interaction space that contains all the required elements to perform the analysis between every pair of cells.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment or a single-cell
        experiment after aggregation into cell types. Columns are
        cell-types/tissues/samples and rows are genes.

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    gene_cutoffs : dict
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

    communication_score : str, default='expression_thresholding'
        Type of communication score used to detect active ligand-receptor
        pairs between each pair of cell. See
        cell2cell.core.communication_scores for more details.
        It can be:

        - 'expression_thresholding'
        - 'expression_product'
        - 'expression_mean'
        - 'expression_gmean'

    cci_score : str, default='bray_curtis'
        Scoring function to aggregate the communication scores. See
        cell2cell.core.cci_scores for more details.
        It can be:

        - 'bray_curtis'
        - 'jaccard'
        - 'count'
        - 'icellnet'

    cci_type : str, default='undirected'
        Type of interaction between two cells. If it is undirected, all ligands
        and receptors are considered from both cells. If it is directed, ligands
        from one cell and receptors from the other are considered separately with
        respect to ligands from the second cell and receptor from the first one.
        So, it can be:

        - 'undirected'
        - 'directed'

    cci_matrix_template : pandas.DataFrame, default=None
        A matrix of shape MxM where M are cell-types/tissues/samples. This
        is used as template for storing CCI scores. It may be useful
        for specifying which pairs of cells to consider.

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

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Attributes
    ----------
    communication_score : str
        Type of communication score used to detect active ligand-receptor
        pairs between each pair of cell. See
        cell2cell.core.communication_scores for more details.
        It can be:

        - 'expression_thresholding'
        - 'expression_product'
        - 'expression_mean'
        - 'expression_gmean'

    cci_score : str
        Scoring function to aggregate the communication scores. See
        cell2cell.core.cci_scores for more details.
        It can be:

        - 'bray_curtis'
        - 'jaccard'
        - 'count'
        - 'icellnet'

    cci_type : str
        Type of interaction between two cells. If it is undirected, all ligands
        and receptors are considered from both cells. If it is directed, ligands
        from one cell and receptors from the other are considered separately with
        respect to ligands from the second cell and receptor from the first one.
        So, it can be:

        - 'undirected'
        - 'directed'

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    modified_rnaseq_data : pandas.DataFrame
        Preprocessed gene expression data for a bulk or single-cell RNA-seq experiment.
        Columns are are cell-types/tissues/samples and rows are genes. The preprocessing
        may correspond to scoring the gene expression as binary or continuous values
        depending on the scoring function for cell-cell interactions/communication.

    interaction_elements : dict
        Dictionary containing all the pairs of cells considered (under
        the key of 'pairs'), Cell instances (under key 'cells')
        which include all cells/tissues/organs with their associated datasets
        (rna_seq, weighted_ppi, etc) and a Cell-Cell Interaction Matrix
        to store CCI scores(under key 'cci_matrix'). A communication matrix
        is also stored in this object when the communication scores are
        computed in the InteractionSpace class (under key
        'communication_matrix')

    distance_matrix : pandas.DataFrame
        Contains distances for each pair of cells, computed from
        the CCI scores previously obtained (and stored in
        interaction_elements['cci_matrix'].
    '''

    def __init__(self, rnaseq_data, ppi_data, gene_cutoffs, communication_score='expression_thresholding',
                 cci_score='bray_curtis', cci_type='undirected', cci_matrix_template=None, complex_sep=None,
                 complex_agg_method='min', interaction_columns=('A', 'B'), verbose=True):

        self.communication_score = communication_score
        self.cci_score = cci_score
        self.cci_type = cci_type

        if self.communication_score == 'expression_thresholding':
            if 'type' in gene_cutoffs.keys():
                cutoff_values = cutoffs.get_cutoffs(rnaseq_data=rnaseq_data,
                                                    parameters=gene_cutoffs,
                                                    verbose=verbose)
            else:
                raise ValueError("If dataframe is not included in gene_cutoffs, please provide the type of method to obtain them.")
        else:
            cutoff_values = None

        prot_a = interaction_columns[0]
        prot_b = interaction_columns[1]
        self.ppi_data = ppi_data.copy()
        if ('A' in self.ppi_data.columns) & (prot_a != 'A'):
            self.ppi_data = self.ppi_data.drop(columns='A')
        if ('B' in self.ppi_data.columns) & (prot_b != 'B'):
            self.ppi_data = self.ppi_data.drop(columns='B')
        self.ppi_data = self.ppi_data.rename(columns={prot_a : 'A', prot_b : 'B'})
        if 'score' not in self.ppi_data.columns:
            self.ppi_data = self.ppi_data.assign(score=1.0)

        self.modified_rnaseq = integrate_data.get_modified_rnaseq(rnaseq_data=rnaseq_data,
                                                                  cutoffs=cutoff_values,
                                                                  communication_score=self.communication_score,
                                                                  )

        self.interaction_elements = generate_interaction_elements(modified_rnaseq=self.modified_rnaseq,
                                                                  ppi_data=self.ppi_data,
                                                                  cci_matrix_template=cci_matrix_template,
                                                                  cci_type=self.cci_type,
                                                                  complex_sep=complex_sep,
                                                                  complex_agg_method=complex_agg_method,
                                                                  verbose=verbose)

        self.interaction_elements['ppi_score'] = self.ppi_data['score'].values

    def pair_cci_score(self, cell1, cell2, cci_score='bray_curtis', use_ppi_score=False, verbose=True):
        '''
        Computes a CCI score for a pair of cells.

        Parameters
        ----------
        cell1 : cell2cell.core.cell.Cell
            First cell-type/tissue/sample to compute the communication
            score. In a directed interaction, this is the sender.

        cell2 : cell2cell.core.cell.Cell
            Second cell-type/tissue/sample to compute the communication
            score. In a directed interaction, this is the receiver.

        cci_score : str, default='bray_curtis'
            Scoring function to aggregate the communication scores between
            a pair of cells. It computes an overall potential of cell-cell
            interactions. If None, it will use the one stored in the
            attribute analysis_setup of this object.
            Options:

            - 'bray_curtis' : Bray-Curtis-like score
            - 'jaccard' : Jaccard-like score
            - 'count' : Number of LR pairs that the pair of cells uses
            - 'icellnet' : Sum of the L-R expression product of a pair of cells

        use_ppi_score : boolean, default=False
            Whether using a weight of LR pairs specified in the ppi_data
            to compute the scores.

        verbose : boolean, default=True
            Whether printing or not steps of the analysis.

        Returns
        -------
        cci_score : float
            Overall score for the interaction between a pair of
            cell-types/tissues/samples. In this case it is a
            Jaccard-like score.
        '''

        if verbose:
            print("Computing interaction score between {} and {}".format(cell1.type, cell2.type))

        if use_ppi_score:
            ppi_score = self.ppi_data['score'].values
        else:
            ppi_score = None
        # Calculate cell-cell interaction score
        if cci_score == 'bray_curtis':
            cci_value = cci_scores.compute_braycurtis_like_cci_score(cell1, cell2, ppi_score=ppi_score)
        elif cci_score == 'jaccard':
            cci_value = cci_scores.compute_jaccard_like_cci_score(cell1, cell2, ppi_score=ppi_score)
        elif cci_score == 'count':
            cci_value = cci_scores.compute_count_score(cell1, cell2, ppi_score=ppi_score)
        elif cci_score == 'icellnet':
            cci_value = cci_scores.compute_icellnet_score(cell1, cell2, ppi_score=ppi_score)
        else:
            raise NotImplementedError("CCI score {} to compute pairwise cell-interactions is not implemented".format(cci_score))
        return cci_value

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

            - 'bray_curtis' : Bray-Curtis-like score
            - 'jaccard' : Jaccard-like score
            - 'count' : Number of LR pairs that the pair of cells uses
            - 'icellnet' : Sum of the L-R expression product of a pair of cells

        use_ppi_score : boolean, default=False
            Whether using a weight of LR pairs specified in the ppi_data
            to compute the scores.

        verbose : boolean, default=True
            Whether printing or not steps of the analysis.

        Returns
        -------
        self.interaction_elements['cci_matrix'] : pandas.DataFrame
            Contains CCI scores for each pair of cells
        '''
        if cci_score is None:
            cci_score = self.cci_score
        else:
            assert isinstance(cci_score, str)

        ### Compute pairwise physical interactions
        if verbose:
            print("Computing pairwise interactions")

        # Compute pair by pair
        for pair in self.interaction_elements['pairs']:
            cell1 = self.interaction_elements['cells'][pair[0]]
            cell2 = self.interaction_elements['cells'][pair[1]]
            cci_value = self.pair_cci_score(cell1,
                                            cell2,
                                            cci_score=cci_score,
                                            use_ppi_score=use_ppi_score,
                                            verbose=verbose)
            self.interaction_elements['cci_matrix'].at[pair[0], pair[1]] = cci_value
            if self.cci_type == 'undirected':
                self.interaction_elements['cci_matrix'].at[pair[1], pair[0]] = cci_value

        # Compute using matmul -> Too slow and uses a lot of memory TODO: Try to optimize this
        # if cci_score == 'bray_curtis':
        #     cci_matrix = cci_scores.matmul_bray_curtis_like(self.interaction_elements['A_score'],
        #                                                     self.interaction_elements['B_score'])
        # self.interaction_elements['cci_matrix'] = pd.DataFrame(cci_matrix,
        #                                                        index=self.interaction_elements['cell_names'],
        #                                                        columns=self.interaction_elements['cell_names']
        #                                                        )

        # Generate distance matrix
        if ~(cci_score in ['count', 'icellnet']):
            self.distance_matrix = self.interaction_elements['cci_matrix'].apply(lambda x: 1 - x)
        else:
            #self.distance_matrix = self.interaction_elements['cci_matrix'].div(self.interaction_elements['cci_matrix'].max().max()).apply(lambda x: 1 - x)
            # Regularized distance
            mean = np.nanmean(self.interaction_elements['cci_matrix'])
            self.distance_matrix = self.interaction_elements['cci_matrix'].div(self.interaction_elements['cci_matrix'] + mean).apply(lambda x: 1 - x)
        np.fill_diagonal(self.distance_matrix.values, 0.0)  # Make diagonal zero (delete autocrine-interactions)

    def pair_communication_score(self, cell1, cell2, communication_score='expression_thresholding',
                                 use_ppi_score=False, verbose=True):
        '''Computes a communication score for each protein-protein interaction
        between a pair of cells.

        Parameters
        ----------
        cell1 : cell2cell.core.cell.Cell
            First cell-type/tissue/sample to compute the communication
            score. In a directed interaction, this is the sender.

        cell2 : cell2cell.core.cell.Cell
            Second cell-type/tissue/sample to compute the communication
            score. In a directed interaction, this is the receiver.

        communication_score : str, default=None
            Type of communication score to infer the potential use of
            a given ligand-receptor pair by a pair of cells/tissues/samples.
            If None, the score stored in the attribute analysis_setup
            will be used.
            Available communication_scores are:

            - 'expression_thresholding' : Computes the joint presence of a
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

        verbose : boolean, default=True
            Whether printing or not steps of the analysis.

        Returns
        -------
        communication_scores : numpy.array
            An array with the communication scores for each intercellular
            PPI.
        '''
        # TODO: Implement communication scores
        if verbose:
            print("Computing communication score between {} and {}".format(cell1.type, cell2.type))

        # Check that new score is the same type as score used to build interaction space (binary or continuous)
        if (communication_score in ['expression_product', 'expression_correlation', 'expression_mean', 'expression_gmean']) \
                & (self.communication_score in ['expression_thresholding', 'differential_combinations']):
            raise ValueError('Cannot use {} for this interaction space'.format(communication_score))
        if (communication_score in ['expression_thresholding', 'differential_combinations']) \
                & (self.communication_score in ['expression_product', 'expression_correlation', 'expression_mean', 'expression_gmean']):
            raise ValueError('Cannot use {} for this interaction space'.format(communication_score))

        if use_ppi_score:
            ppi_score = self.ppi_data['score'].values
        else:
            ppi_score = None

        if communication_score in ['expression_thresholding', 'differential_combinations']:
            communication_value = communication_scores.get_binary_scores(cell1=cell1,
                                                                         cell2=cell2,
                                                                         ppi_score=ppi_score)
        elif communication_score in ['expression_product', 'expression_correlation', 'expression_mean', 'expression_gmean']:
              communication_value = communication_scores.get_continuous_scores(cell1=cell1,
                                                                               cell2=cell2,
                                                                               ppi_score=ppi_score,
                                                                               method=communication_score)
        else:
            raise NotImplementedError(
                "Communication score {} to compute pairwise cell-communication is not implemented".format(communication_score))
        return communication_value

    def compute_pairwise_communication_scores(self, communication_score=None, use_ppi_score=False, ref_ppi_data=None,
                                              interaction_columns=('A', 'B'), cells=None, cci_type=None, verbose=True):
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

            - 'expression_thresholding' : Computes the joint presence of a
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
            will be used

        cells : list=None
            List of cells to consider.

        cci_type : str, default=None
            Type of interaction between two cells. Used to specify
            if we want to consider a LR pair in both directions.
            It can be:
                - 'undirected'
                - 'directed
            If None, the one stored in the attribute analysis_setup
            will be used.

        verbose : boolean, default=True
            Whether printing or not steps of the analysis.

        Returns
        -------
        self.interaction_elements['communication_matrix'] : pandas.DataFrame
            Contains communication scores for each LR pair in a
            given pair of sender-receiver cells.
        '''
        if communication_score is None:
            communication_score = self.communication_score
        else:
            assert isinstance(communication_score, str)

        # Cells to consider
        if cells is None:
            cells = self.interaction_elements['cell_names']

        # Labels:
        if cci_type is None:
            cell_pairs = self.interaction_elements['pairs']
        elif cci_type != self.cci_type:
            cell_pairs = generate_pairs(cells, cci_type)
        else:
            #cell_pairs = generate_pairs(cells, self.cci_type) # Think about other scenarios that may need this line
            cell_pairs = self.interaction_elements['pairs']
        col_labels = ['{};{}'.format(pair[0], pair[1]) for pair in cell_pairs]

        # Ref PPI data
        if ref_ppi_data is None:
            ref_index = self.ppi_data.apply(lambda row: (row['A'], row['B']), axis=1)
            keep_index = list(range(self.ppi_data.shape[0]))
        else:
            ref_ppi = ref_ppi_data.copy()
            prot_a = interaction_columns[0]
            prot_b = interaction_columns[1]
            if ('A' in ref_ppi.columns) & (prot_a != 'A'):
                ref_ppi = ref_ppi.drop(columns='A')
            if ('B' in ref_ppi.columns) & (prot_b != 'B'):
                ref_ppi = ref_ppi.drop(columns='B')
            ref_ppi = ref_ppi.rename(columns={prot_a: 'A', prot_b: 'B'})
            ref_index = list(ref_ppi.apply(lambda row: (row['A'], row['B']), axis=1).values)
            keep_index = list(pd.merge(self.ppi_data, ref_ppi, how='inner').index)

        # DataFrame to Store values
        communication_matrix = pd.DataFrame(index=ref_index, columns=col_labels)

        ### Compute pairwise physical interactions
        if verbose:
            print("Computing pairwise communication")

        for i, pair in enumerate(cell_pairs):
            cell1 = self.interaction_elements['cells'][pair[0]]
            cell2 = self.interaction_elements['cells'][pair[1]]

            comm_score = self.pair_communication_score(cell1,
                                                       cell2,
                                                       communication_score=communication_score,
                                                       use_ppi_score=use_ppi_score,
                                                       verbose=verbose)
            kept_values = comm_score.flatten()[keep_index]
            communication_matrix[col_labels[i]] = kept_values

        self.interaction_elements['communication_matrix'] = communication_matrix