# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.preprocessing import integrate_data, cutoffs
from cell2cell.core import cell, cci_scores

import itertools
import numpy as np
import pandas as pd


def generate_interaction_elements(modified_rnaseq, ppi_data, interaction_type, cci_matrix_template=None, verbose=True):
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
        print('Creating ' + interaction_type.capitalize() + ' Interaction Space')

    # Cells
    cell_instances = list(modified_rnaseq.columns)  # @Erick, check if position 0 of columns contain index header.
    cell_number = len(cell_instances)

    # Generate pairwise interactions
    pairwise_interactions = list(itertools.combinations(cell_instances + cell_instances, 2))
    pairwise_interactions = list(set(pairwise_interactions))  # Remove duplicates

    # Interaction elements
    interaction_space = {}
    interaction_space['pairs'] = pairwise_interactions
    interaction_space['cells'] = cell.get_cells_from_rnaseq(modified_rnaseq, verbose=verbose)

    # Cell-specific binary ppi
    for cell_instance in interaction_space['cells'].values():
        cell_instance.weighted_ppi = integrate_data.get_weighted_ppi(ppi_data=ppi_data,
                                                                     modified_rnaseq_data=cell_instance.rnaseq_data,
                                                                     column='value')

    # Cell-cell interaction matrix
    if cci_matrix_template is None:
        interaction_space['cci_matrix'] = pd.DataFrame(np.zeros((cell_number, cell_number)),
                                                       columns=cell_instances,
                                                       index=cell_instances)
    else:
        interaction_space['cci_matrix'] = cci_matrix_template

    return interaction_space


class InteractionSpace():
    '''
    Interaction space that contains all the required elements to perform the analysis between every pair of cells.

    Parameters
    ----------


    Attributes
    ----------


    '''

    def __init__(self, rnaseq_data, ppi_dict, interaction_type, gene_cutoffs, score_type='binary',
                 cci_matrix_template=None, verbose=True):

        self.score_type = score_type
        if self.score_type == 'binary':
            if 'type' in gene_cutoffs.keys():
                cutoff_values = cutoffs.get_cutoffs(rnaseq_data=rnaseq_data,
                                                    parameters=gene_cutoffs,
                                                    verbose=verbose)
            else:
                raise ValueError("If dataframe is not included in gene_cutoffs, please provide the type of method to obtain them.")
        else:
            cutoff_values = None

        self.interaction_type = interaction_type
        self.ppi_data = ppi_dict[self.interaction_type]

        self.modified_rnaseq = integrate_data.get_modified_rnaseq(rnaseq_data=rnaseq_data,
                                                                  score_type=self.score_type,
                                                                  cutoffs=cutoff_values)

        self.interaction_elements = generate_interaction_elements(modified_rnaseq=self.modified_rnaseq,
                                                                  ppi_data=self.ppi_data,
                                                                  interaction_type=self.interaction_type,
                                                                  cci_matrix_template=cci_matrix_template,
                                                                  verbose=verbose)

    def pairwise_interaction(self, cell1, cell2, score_metric='bray_curtis', verbose=True):
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
            print("Computing {} interaction between {} and {}".format(self.interaction_type, cell1.type, cell2.type))

        # Calculate cell-cell interaction score
        if score_metric == 'bray_curtis':
            cci_score = cci_scores.compute_braycurtis_like_cci_score(cell1, cell2)
        elif score_metric == 'jaccard':
            cci_score = cci_scores.compute_jaccard_like_cci_score(cell1, cell2)
        else:
            raise NotImplementedError("Score metric {} to compute pairwise cell-interactions is not implemented".format(score_metric))
        return cci_score

    def compute_pairwise_interactions(self, score_metric='bray_curtis', verbose=True):
        '''
        Function that computes...

        Parameters
        ----------


        Returns
        -------


        '''
        ### Compute pairwise physical interactions
        if verbose:
            print("Computing pairwise interactions")
        for pair in self.interaction_elements['pairs']:
            cell1 = self.interaction_elements['cells'][pair[0]]
            cell2 = self.interaction_elements['cells'][pair[1]]
            cci_score = self.pairwise_interaction(cell1,
                                                  cell2,
                                                  score_metric=score_metric,
                                                  verbose=verbose)
            self.interaction_elements['cci_matrix'].loc[pair[0], pair[1]] = cci_score
            self.interaction_elements['cci_matrix'].loc[pair[1], pair[0]] = cci_score