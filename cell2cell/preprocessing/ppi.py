# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.utils import resample


### Manipulate PPI networks
def remove_ppi_bidirectionality(PPI_data, interaction_columns, verbose=True):
    '''
    This function removes duplicate interactions. For example, when P1-P2 and P2-P1 interactions are present in the
    dataset, only one of them will remain.
    '''
    if verbose:
        print('Removing bidirectionality of PPI network')
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]
    IA = PPI_data[[header_interactorA, header_interactorB]]
    IB = PPI_data[[header_interactorB, header_interactorA]]
    IB.columns = [header_interactorA, header_interactorB]
    repeated_interactions = pd.merge(IA, IB, on=[header_interactorA, header_interactorB])
    repeated = list(np.unique(repeated_interactions.values.flatten()))
    df =  pd.DataFrame(combinations(sorted(repeated), 2), columns=[header_interactorA, header_interactorB])
    df = df[[header_interactorB, header_interactorA]]   # To keep lexicographically sorted interactions
    df.columns = [header_interactorA, header_interactorB]
    unidirectional_ppi = pd.merge(PPI_data, df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    unidirectional_ppi.reset_index(drop=True, inplace=True)
    return unidirectional_ppi


def simplify_ppi(ppi_data, interaction_columns, score=None, verbose=True):
    if verbose:
        print('Simplying PPI network')
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]
    if score is None:
        simple_ppi = ppi_data[[header_interactorA, header_interactorB]]
        simple_ppi = simple_ppi.assign(score = 1.0)
    else:
        simple_ppi = ppi_data[[header_interactorA, header_interactorB, score]]
        simple_ppi[score] = simple_ppi[score].fillna(np.nanmin(simple_ppi[score].values))
    cols = ['A', 'B', 'score']
    simple_ppi.columns = cols
    return simple_ppi


def filter_ppi_by_proteins(ppi_data, proteins):
    ppi_data = ppi_data[ppi_data['A'].isin(proteins)]
    ppi_data = ppi_data[ppi_data['B'].isin(proteins)]
    ppi_data = ppi_data.reset_index(drop=True)
    return ppi_data


def bidirectional_ppi_for_cci(ppi_data, interaction_columns=['A', 'B'], verbose=True):
    if verbose:
        print("Making bidirectional PPI for CCI.")
    if ppi_data.shape[0] == 0:
        return ppi_data.copy()

    ppi_A = ppi_data.copy()
    col1 = ppi_A[[interaction_columns[0]]]
    col2 = ppi_A[[interaction_columns[1]]]
    ppi_B = ppi_data.copy()
    ppi_B[interaction_columns[0]] = col2
    ppi_B[interaction_columns[1]] = col1

    if verbose:
        print("Removing duplicates in bidirectional PPI network.")
    new_ppi_data = pd.concat([ppi_A, ppi_B], join="inner")
    new_ppi_data = new_ppi_data.drop_duplicates()
    new_ppi_data.reset_index(inplace=True, drop=True)
    return new_ppi_data


def filter_ppi_network(ppi_data, contact_proteins, mediator_proteins=None, reference_list=None,
                       interaction_type='contacts', interaction_columns=['A', 'B'], verbose=True):
    '''
    :param ppi_data:
    :param interaction_columns:
    :param contact_proteins: List of genes/proteins consistent with IDs used in PPI
    :return:
    '''

    new_ppi_data = get_filtered_ppi_network(ppi_data=ppi_data,
                                            contact_proteins=contact_proteins,
                                            mediator_proteins=mediator_proteins,
                                            reference_list=reference_list,
                                            interaction_type=interaction_type,
                                            interaction_columns=interaction_columns,
                                            verbose=verbose)

    new_ppi_data = bidirectional_ppi_for_cci(ppi_data=new_ppi_data,
                                             interaction_columns=interaction_columns,
                                             verbose=verbose)
    return new_ppi_data


def get_filtered_ppi_network(ppi_data, contact_proteins, mediator_proteins=None, reference_list=None,
                             interaction_type='contacts', interaction_columns=['A', 'B'], verbose=True):
    if (mediator_proteins is None) and (interaction_type != 'contacts'):
        raise ValueError("mediator_proteins cannot be None when interaction_type is not contacts")

    # Consider only genes that are in the reference_list:
    if reference_list is not None:
        contact_proteins = list(filter(lambda x: x in reference_list , contact_proteins))
        if mediator_proteins is not None:
            mediator_proteins = list(filter(lambda x: x in reference_list, mediator_proteins))

    if verbose:
        print('Filtering PPI interactions by using a list of genes for {} interactions'.format(interaction_type))
    if interaction_type == 'contacts':
        new_ppi_data = get_all_to_all_ppi(ppi_data=ppi_data,
                                          proteins=contact_proteins,
                                          interaction_columns=interaction_columns)

    elif interaction_type == 'complete':
        total_proteins = list(set(contact_proteins + mediator_proteins))

        new_ppi_data = get_all_to_all_ppi(ppi_data=ppi_data,
                                          proteins=total_proteins,
                                          interaction_columns=interaction_columns)
    else:
        # All the following interactions incorporate contacts-mediator interactions
        mediated = get_one_group_to_other_ppi(ppi_data=ppi_data,
                                              proteins_a=contact_proteins,
                                              proteins_b=mediator_proteins,
                                              interaction_columns=interaction_columns)
        if interaction_type == 'mediated':
            new_ppi_data = mediated
        elif interaction_type == 'combined':
            contacts = get_all_to_all_ppi(ppi_data=ppi_data,
                                          proteins=contact_proteins,
                                          interaction_columns=interaction_columns)
            new_ppi_data = pd.concat([contacts, mediated], ignore_index = True).drop_duplicates()
        else:
            raise NameError('Not valid interaction type to filter the PPI network')
    new_ppi_data.reset_index(inplace=True, drop=True)
    return new_ppi_data


def get_all_to_all_ppi(ppi_data, proteins, interaction_columns=['A', 'B']):
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]
    new_ppi_data = ppi_data.loc[ppi_data[header_interactorA].isin(proteins) & ppi_data[header_interactorB].isin(proteins)]
    new_ppi_data = new_ppi_data.drop_duplicates()
    return new_ppi_data


def get_one_group_to_other_ppi(ppi_data, proteins_a, proteins_b, interaction_columns=['A', 'B']):
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]
    direction1 = ppi_data.loc[ppi_data[header_interactorA].isin(proteins_a) & ppi_data[header_interactorB].isin(proteins_b)]
    direction2 = ppi_data.loc[ppi_data[header_interactorA].isin(proteins_b) & ppi_data[header_interactorB].isin(proteins_a)]
    direction2.columns = [header_interactorB, header_interactorA, 'score']
    direction2 = direction2[[header_interactorA, header_interactorB, 'score']]
    new_ppi_data = pd.concat([direction1, direction2], ignore_index=True).drop_duplicates()
    return new_ppi_data