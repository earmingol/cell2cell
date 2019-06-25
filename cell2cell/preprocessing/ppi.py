# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd

from itertools import combinations


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
    cols = ['A', 'B', 'score']
    simple_ppi.columns = cols
    return simple_ppi


def ppi_name_match(ppi_data, names):
    ppi_data = ppi_data[ppi_data['A'].isin(names)]
    ppi_data = ppi_data[ppi_data['B'].isin(names)]
    ppi_data = ppi_data.reset_index(drop=True)
    return ppi_data


def filter_ppi_network(ppi_data, contact_proteins, mediator_proteins=None, reference_list=None,
                       interaction_type='contacts', interaction_columns=['A', 'B'], verbose=True):
    '''
    :param ppi_data:
    :param interaction_columns:
    :param contact_proteins: List of genes/proteins consistent with IDs used in PPI
    :return:
    '''
    if (mediator_proteins is None) and (interaction_type != 'contacts'):
        raise ValueError("mediator_proteins cannot be None when interaction_type is not contacts")

    # Consider only genes that are in the reference_list:
    if reference_list is not None:
        contact_proteins = list(filter(lambda x: x in reference_list , contact_proteins))
        if mediator_proteins is not None:
            mediator_proteins = list(filter(lambda x: x in reference_list, mediator_proteins))

    # PPI Headers
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]

    # Contact-Contact interactions
    contacts = ppi_data.loc[ppi_data[header_interactorA].isin(contact_proteins) & ppi_data[header_interactorB].isin(contact_proteins)]
    if verbose:
        print('Filtering PPI interactions by using a list of genes for {} interactions'.format(interaction_type))
    if interaction_type == 'contacts':
        new_ppi_data = contacts.drop_duplicates()
    else:
        # Extracell-surface interactions
        mediated1 = ppi_data.loc[ppi_data[header_interactorA].isin(contact_proteins) & ppi_data[header_interactorB].isin(mediator_proteins)]
        mediated2 = ppi_data.loc[ppi_data[header_interactorA].isin(mediator_proteins) & ppi_data[header_interactorB].isin(contact_proteins)]
        mediated2.columns = [header_interactorB, header_interactorA, 'score']
        mediated2 = mediated2[[header_interactorA, header_interactorB, 'score']]
        mediated = pd.concat([mediated1, mediated2], ignore_index = True).drop_duplicates()
        if interaction_type == 'mediated':
            new_ppi_data = mediated
        elif interaction_type == 'combined':
            new_ppi_data = pd.concat([contacts, mediated], ignore_index = True).drop_duplicates()
        else:
            raise NameError('Not valid interaction type to filter the PPI network')
    return new_ppi_data
