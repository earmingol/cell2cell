# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


def compute_ppi_score(abundance_cell1, abundance_cell2, score):
    '''Function that calculate an score for the interaction of a given pair of proteins based on RNA abundance.
    This score is based on ........

    Parameters
    ----------
    abundance_cell1 : pandas DataFrame object
        Abundance of respective gene/protein calculated from RNAseq data for cell1.

    abundance_cell2 : pandas DataFrame object
        Abundance of respective gene/protein calculated from RNAseq data for cell2.

    score : float
        Respective score of the interaction between gene/protein in cell1 and gene/protein in cell2.

    Returns
    -------
    ppi_score : float
        Score for the interaction of the gene/protein of the pair of cells based on their relative abundances.
    '''
    ppi_score = score * abundance_cell1 * abundance_cell2
    return ppi_score


def compute_cci_score(cell1, cell2, interaction_matrix, PPI_data):
    '''Function that calculate an score for the interaction between two cells based on the interactions of their proteins
    with the proteins of the other cell.
    This score is based on ........

    Parameters
    ----------
    interaction_matrix : pandas DataFrame
        Matrix that contains the interaction scores between all the proteins of one cell with the proteins of the other cell.

    Returns
    -------
    cci_score : float
        Score for the interaction of the gene/protein of the pair of cells based on their relative abundances.
    '''
    cell1_total_interactions = 0
    cell2_total_interactions = 0
    for index, row in PPI_data.iterrows():
        try:  # To avoid errors when a gene in PPI is not in RNAseq data
            A = row['A']
            B = row['B']
            cell1_total_interactions += cell1.relative_abundance['value'].loc[A] + \
                                        cell1.relative_abundance['value'].loc[B]
            cell2_total_interactions += cell2.relative_abundance['value'].loc[A] + \
                                        cell2.relative_abundance['value'].loc[B]
        except:
            pass
    #### Number of interactions between cells / Sum of potential interaction of both cells
    numerator = interaction_matrix.values.sum() + np.trace(interaction_matrix.values)
    denominator = (cell1_total_interactions + cell2_total_interactions) / 2.0
    cci_score = numerator / denominator

    if cci_score is np.nan:
        cci_score = 0.0
    return cci_score