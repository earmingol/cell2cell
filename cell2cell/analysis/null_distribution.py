# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd

from sklearn.utils import resample, shuffle

from cell2cell.preprocessing import ppi


### FUNCTIONS FOR GENERATING DIFFERENT KIND OF NULL DISTIBUTIONS ###
def filter_null_ppi_network(ppi_data, contact_proteins, mediator_proteins=None, interaction_type='contacts',
                            interaction_columns=['A', 'B'], randomized_col=0, random_state=None, verbose=True):
    '''
    This function
    '''

    new_ppi_data = get_null_ppi_network(ppi_data=ppi_data,
                                        contact_proteins=contact_proteins,
                                        mediator_proteins=mediator_proteins,
                                        interaction_type=interaction_type,
                                        interaction_columns=interaction_columns,
                                        randomized_col=randomized_col,
                                        random_state=random_state,
                                        verbose=verbose)

    new_ppi_data = ppi.bidirectional_ppi_for_cci(ppi_data=new_ppi_data,
                                                 interaction_columns=interaction_columns,
                                                 verbose=verbose)
    return new_ppi_data


def filter_random_subsample_ppi_network(original_ppi_data, contact_proteins, mediator_proteins=None,
                                        interaction_type='contacts', random_ppi_data=None, interaction_columns=['A', 'B'],
                                        random_state=None, verbose=True):
    '''
    This function randomly subsample a ppi network, if a random_ppi_data is provided, it is subsampled instead.
    '''

    new_ppi_data = ppi.get_filtered_ppi_network(ppi_data=original_ppi_data,
                                                contact_proteins=contact_proteins,
                                                mediator_proteins=mediator_proteins,
                                                interaction_type=interaction_type,
                                                interaction_columns=interaction_columns,
                                                verbose=verbose)

    if random_ppi_data is not None:
        new_random = ppi.get_filtered_ppi_network(ppi_data=random_ppi_data,
                                                 contact_proteins=contact_proteins,
                                                 mediator_proteins=mediator_proteins,
                                                 interaction_type=interaction_type,
                                                 interaction_columns=interaction_columns,
                                                 verbose=verbose)

        new_ppi_data = subsample_dataframe(df=new_random,
                                           n_samples=new_ppi_data.shape[0],
                                           random_state=random_state)
    else:
        #new_ppi_data = shuffle_dataframe(df=new_ppi_data,
        #                                 shuffling_number=1,
        #                                 axis=0, random_state=random_state)
        new_ppi_data = subsample_dataframe(df=new_ppi_data,
                                           n_samples=new_ppi_data.shape[0],
                                           random_state=random_state)

    new_ppi_data = ppi.bidirectional_ppi_for_cci(ppi_data=new_ppi_data,
                                                 interaction_columns=interaction_columns,
                                                 verbose=verbose)
    return new_ppi_data


def get_null_ppi_network(ppi_data, contact_proteins, mediator_proteins=None, interaction_type='contacts',
                         interaction_columns=['A', 'B'], randomized_col=0, random_state=None, verbose=True):
    if (mediator_proteins is None) and (interaction_type != 'contacts'):
        raise ValueError("mediator_proteins cannot be None when interaction_type is not contacts")

    if verbose:
        print('Filtering PPI interactions by using a list of genes for {} interactions'.format(interaction_type))
    if interaction_type == 'contacts':
        new_ppi_data = ppi.get_all_to_all_ppi(ppi_data=ppi_data,
                                              proteins=contact_proteins,
                                              interaction_columns=interaction_columns)

        in_contacts = find_elements_in_dataframe(df=new_ppi_data,
                                                 elements=contact_proteins,
                                                 columns=interaction_columns)

        new_ppi_data[interaction_columns[randomized_col]] = resample(in_contacts,
                                                                     n_samples=new_ppi_data.shape[0],
                                                                     random_state=random_state)
    elif interaction_type == 'complete':
        total_proteins = list(set(contact_proteins + mediator_proteins))

        new_ppi_data = ppi.get_all_to_all_ppi(ppi_data=ppi_data,
                                              proteins=total_proteins,
                                              interaction_columns=interaction_columns)

        in_total = find_elements_in_dataframe(df=new_ppi_data,
                                              elements=total_proteins,
                                              columns=interaction_columns)

        new_ppi_data[interaction_columns[randomized_col]] = resample(in_total,
                                                                     n_samples=new_ppi_data.shape[0],
                                                                     random_state=random_state)

    else:
        # All the following interactions incorporate contacts-mediator interactions
        mediated = ppi.get_one_group_to_other_ppi(ppi_data=ppi_data,
                                                  proteins_a=contact_proteins,
                                                  proteins_b=mediator_proteins,
                                                  interaction_columns=interaction_columns)

        in_mediated = find_elements_in_dataframe(df=mediated,
                                                 elements=mediator_proteins,
                                                 columns=interaction_columns)

        mediated[interaction_columns[randomized_col]] = resample(in_mediated,
                                                                 n_samples=mediated.shape[0],
                                                                 random_state=random_state)

        if interaction_type == 'mediated':
            new_ppi_data = mediated
        elif interaction_type == 'combined':
            contacts = ppi.get_all_to_all_ppi(ppi_data=ppi_data,
                                              proteins=contact_proteins,
                                              interaction_columns=interaction_columns)

            in_contacts = find_elements_in_dataframe(df=contacts,
                                                     elements=contact_proteins,
                                                     columns=interaction_columns)

            contacts[interaction_columns[randomized_col]] = resample(in_contacts,
                                                                     n_samples=contacts.shape[0],
                                                                     random_state=random_state)

            new_ppi_data = pd.concat([contacts, mediated], ignore_index=True).drop_duplicates()
        else:
            raise NameError('Not valid interaction type to filter the PPI network')
    new_ppi_data.reset_index(inplace=True, drop=True)
    return new_ppi_data


def find_elements_in_dataframe(df, elements, columns=None):
    if columns is None:
        columns = list(df.columns)
    df_elements = pd.Series(np.unique(df[columns].values.flatten()))
    df_elements = df_elements.loc[df_elements.isin(elements)].values
    return list(df_elements)


def subsample_dataframe(df, n_samples, random_state=None):
    subsampled_df = df.sample(n_samples, random_state=random_state).reset_index(drop=True)
    return subsampled_df


def random_switching_ppi_labels(interaction_type, ppi_dict, genes=None, random_state=None, interaction_columns=['A', 'B']):
    ppi_dict_ = dict()
    ppi_dict_[interaction_type] = ppi_dict[interaction_type].copy()
    if genes is None:
        genes = list(np.unique(ppi_dict_[interaction_type][interaction_columns].values.flatten()))
    else:
        genes = list(set(genes))
    mapper = dict(zip(genes, shuffle(genes, random_state=random_state)))
    A = interaction_columns[0]
    B = interaction_columns[1]
    ppi_dict_[interaction_type][A] = ppi_dict_[interaction_type][A].apply(lambda x: mapper[x])
    ppi_dict_[interaction_type][B] = ppi_dict_[interaction_type][B].apply(lambda x: mapper[x])
    return mapper, ppi_dict_


def shuffle_dataframe(df, shuffling_number=1, axis=0, random_state=None):
    df_ = df.copy()
    axis = int(not axis)  # pandas.DataFrame is always 2D
    for _ in range(shuffling_number):
        for i, view in enumerate(np.rollaxis(df_.values, axis)):
            if random_state is not None:
                np.random.seed(random_state + i)
            np.random.shuffle(view)
    return df_