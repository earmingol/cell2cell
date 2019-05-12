# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.preprocessing.binary import get_binary_rnaseq

import itertools
import numpy as np
import pandas as pd


def generate_interaction_elements(binary_data, interaction_type, verbose=True):
    '''Create all pointers of pairwise interactions needed to perform the analyses.
    All these information as well as the related data to each pairwise interaction is called Interaction Space.

    Parameters
    ----------



    Returns
    -------
    interaction_space : dict
        Dictionary containing all the pairwise interaction ('pairs' key), product available to interact ('product_available'),
        cell-cell protein interaction matrices ('interaction_matrices' key) and Cell-Cell Interaction Matrix that scores
        the possible interaction between each pair of cells ('CCI' key).
    '''

    if verbose:
        print('Creating ' + interaction_type.capitalize() + ' Interaction Space')

    # Cells
    cell_instances = list(binary_data.columns)  # @Erick, check if position 0 of columns contain index header.
    cell_number = len(cell_instances)

    # Generate pairwise interactions
    pairwise_interactions = list(itertools.combinations(cell_instances + cell_instances, 2))
    pairwise_interactions = list(set(pairwise_interactions))  # Remove duplicates

    # Interaction elements
    interaction_space = {}
    interaction_space['pairs'] = pairwise_interactions
    interaction_space['interaction_matrices'] = {}

    # Cell-cell interaction matrix
    interaction_space['cci_matrix'] = pd.DataFrame(np.zeros((cell_number, cell_number)), columns=cell_instances, index=cell_instances)

    return interaction_space


class InteractionSpace():
    def __init__(self, rnaseq_data, ppi_data, interaction_type, gene_cutoffs, verbose = True):
        self.binary_data = get_binary_rnaseq(rnaseq_data, gene_cutoffs)
        self.ppi_data = ppi_data
        self.interaction_type = interaction_type
        self.interaction_space = generate_interaction_elements(self.binary_data, self.interaction_type, verbose)


    def pairwise_interaction(self, cell1, cell2, ppi_data, verbose=True):
        '''Function that perform the interaction analysis of a pair of cells.

        Parameters
        ----------
        cell1 : Cell class
            Cell instance generated from RNAseq data.

        cell2 : Cell class
            Cell instance generated from RNAseq data.

        ppi_data: pandas DataFrame object
            DataFrame containing three columns. The first and second ones are for proteins, where each row is an interaction.
            The third column is the 'score' or probability for such interaction. This data should be processed before with 'simplify_ppi'
            in order to reduce the table to the three columns mentioned before.

        product_filter : list or None (default = None)
            List of gene names to consider based on a previous filter with GO terms or any other common pattern. Only those
            genes will be considered and the others excluded.

        Returns
        -------
        cci_score : float
            Score for a the interaction of the specified pair of cells. This score is computed considering all the inter-cell protein-protein
            interactions of the pair of cells.

        interaction_matrix : pandas DataFrame object
            Matrix that contains the respective score for the interaction between a pair of proteins (one protein on cell1
            and the other on cell2).
        '''

        if verbose:
            print("Computing physical interaction between " + cell1.type + " and " + cell2.type)

        # Creating protein-protein interaction matrix
        product_filter = list(set(list(ppi_data['A'].values) + list(ppi_data['B'].values)))
        product_number = len(product_filter)
        data = np.zeros((product_number, product_number))
        interaction_matrix = pd.DataFrame(data, columns=product_filter, index=product_filter)

        # Filling out protein-protein interaction matrix
        for index, row in ppi_data.iterrows():  ###### MAKE THIS FOR PARALLEL OR THE OTHER WHEN RUNNING ALL PAIRWISE ANALYSES (CElegans_analysis.py)?
            # Use function to evaluate PPI based on score and RNAseq_data
            try:  # To avoid errors when a gene in PPI is not in RNAseq data
                A = row['A']
                B = row['B']
                score = row['score']
                interaction_matrix.loc[A, B] = compute_ppi_score(cell1.relative_abundance['value'].loc[A],
                                                                 cell2.relative_abundance['value'].loc[B], score)
                interaction_matrix.loc[B, A] = compute_ppi_score(cell1.relative_abundance['value'].loc[B],
                                                                 cell2.relative_abundance['value'].loc[A], score)
            except:
                pass

        # Calculate cell-cell interaction score
        cci_score = compute_cci_score(cell1, cell2, interaction_matrix,
                                      ppi_data)  # Function to compute cell-cell interaction score
        return cci_score, interaction_matrix

    def compute_pairwise_interactions(self, verbose = True):
        ### Compute pairwise physical interactions
        print("Computing pairwise physical interactions")
        for pair in self.interaction_space['pairs']:  ###### MAKE THIS FOR PARALLEL?
            cell1 = self.cells[pair[0]]
            cell2 = self.cells[pair[1]]
            cci_score, interaction_matrix = self.pairwise_interaction(cell1, cell2, self.PPI_data, verbose = verbose)
            self.interaction_space['interaction_matrices'][pair] = interaction_matrix
            self.interaction_space['CCI_matrix'].loc[pair[0], pair[1]] = cci_score
            self.interaction_space['CCI_matrix'].loc[pair[1], pair[0]] = cci_score

    def get_genes_for_filter(self):
        genes = []
        if self.product_filter is not None:
            genes += self.product_filter
        if self.mediators is not None:
            genes += self.mediators
        genes = list(set(genes))
        return genes