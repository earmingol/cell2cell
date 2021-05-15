# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.preprocessing import integrate_data, cutoffs, get_genes_from_complexes, add_complexes_to_expression
from cell2cell.core import cell, cci_scores, communication_scores

import itertools
import numpy as np
import pandas as pd


def generate_pairs(cells, cci_type, self_interaction=True, remove_duplicates=True):
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
                                  complex_sep=None, interaction_columns=('A', 'B'), verbose=True):
    '''Create all elements of pairwise interactions needed to perform the analyses.
    All these variables are used by the class InteractionSpace.

    Parameters
    ----------



    Returns
    -------
    interaction_space : dict
        Dictionary containing all the pairwise interaction ('pairs' key), cell dictionary ('cells' key) containing all
        cells/tissues/organs with their associated datasets (rna_seq, binary_ppi, etc) and a Cell-Cell Interaction Matrix
        that scores the possible interaction between each pair of cells ('cci_matrix key) for the respective interaction
        type.
    '''

    if verbose:
        print('Creating Interaction Space')

    # Include complex expression
    if complex_sep is not None:
        col_a_genes, complex_a, col_b_genes, complex_b, complexes = get_genes_from_complexes(ppi_data=ppi_data,
                                                                                             complex_sep=complex_sep,
                                                                                             interaction_columns=interaction_columns
                                                                                             )
        modified_rnaseq = add_complexes_to_expression(modified_rnaseq, complexes)

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


    Attributes
    ----------


    '''

    def __init__(self, rnaseq_data, ppi_data, gene_cutoffs, communication_score='expression_thresholding',
                 cci_score='bray_curtis', cci_type='undirected', cci_matrix_template=None, complex_sep=None,
                 interaction_columns=('A', 'B'), verbose=True):

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
                                                                  communication_score=self.communication_score,
                                                                  cutoffs=cutoff_values
                                                                  )

        self.interaction_elements = generate_interaction_elements(modified_rnaseq=self.modified_rnaseq,
                                                                  ppi_data=self.ppi_data,
                                                                  cci_matrix_template=cci_matrix_template,
                                                                  cci_type=self.cci_type,
                                                                  complex_sep=complex_sep,
                                                                  verbose=verbose)

        self.interaction_elements['ppi_score'] = self.ppi_data['score'].values

    def pair_cci_score(self, cell1, cell2, cci_score='bray_curtis', use_ppi_score=False, verbose=True):
        '''
        Function that performs the interaction analysis of a pair of cells.

        Parameters
        ----------
        cell1 : Cell class
            Cell instance generated from RNAseq datasets.

        cell2 : Cell class
            Cell instance generated from RNAseq datasets.

        type : str, 'binary' by default
            Function type to compute the interaction. In this case calculation based on binary gene expressions is perform.

        verbose : boolean, True by default
            It shows printings of function if True.

        Returns
        -------
        cci_score : float
            Score for a the interaction of the specified pair of cells. This score is computed considering all the inter-cell protein-protein
            interactions of the pair of cells.
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
        else:
            raise NotImplementedError("CCI score {} to compute pairwise cell-interactions is not implemented".format(cci_score))
        return cci_value

    def compute_pairwise_cci_scores(self, cci_score=None, use_ppi_score=False, verbose=True):
        '''
        Function that computes...

        Parameters
        ----------


        Returns
        -------


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
        if cci_score != 'count':
            self.distance_matrix = self.interaction_elements['cci_matrix'].apply(lambda x: 1 - x)
        else:
            self.distance_matrix = self.interaction_elements['cci_matrix'].div(self.interaction_elements['cci_matrix'].max().max()).apply(lambda x: 1 - x)
        np.fill_diagonal(self.distance_matrix.values, 0.0)  # Make diagonal zero (delete autocrine-interactions)

    def pair_communication_score(self, cell1, cell2, communication_score='expression_thresholding',
                                 use_ppi_score=False, verbose=True):
        # TODO: Implement communication scores
        if verbose:
            print("Computing communication score between {} and {}".format(cell1.type, cell2.type))

        # Check that new score is the same type as score used to build interaction space (binary or continuous)
        if (communication_score in ['expression_product', 'expression_correlation', 'expression_mean']) \
                & (self.communication_score in ['expression_thresholding', 'differential_combinations']):
            raise ValueError('Cannot use {} for this interaction space'.format(communication_score))
        if (communication_score in ['expression_thresholding', 'differential_combinations']) \
                & (self.communication_score in ['expression_product', 'expression_correlation', 'expression_mean']):
            raise ValueError('Cannot use {} for this interaction space'.format(communication_score))

        if use_ppi_score:
            ppi_score = self.ppi_data['score'].values
        else:
            ppi_score = None

        if (communication_score == 'expression_thresholding') | (communication_score == 'differential_combinations'):
            communication_value = communication_scores.get_binary_scores(cell1=cell1,
                                                                         cell2=cell2,
                                                                         ppi_score=ppi_score)
        elif (communication_score == 'expression_product') | (communication_score == 'expression_correlation') \
                | (communication_score == 'expression_mean'):
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
        '''
        Function that computes...

        Parameters
        ----------


        Returns
        -------


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