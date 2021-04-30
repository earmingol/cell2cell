# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd

from itertools import combinations


### Preprocess a PPI table from a known list
def preprocess_ppi_data(ppi_data, interaction_columns, sort_values=None, score=None, rnaseq_genes=None, complex_sep=None,
             dropna=False, strna='', upper_letter_comparison=True, verbose=True):
    if sort_values is not None:
        ppi_data = ppi_data.sort_values(by=sort_values)
    if dropna:
        ppi_data = ppi_data.loc[ppi_data[interaction_columns].dropna().index,:]
    if strna is not None:
        assert(isinstance(strna, str)), "strna has to be an string."
        ppi_data = ppi_data.fillna(strna)
    unidirectional_ppi = remove_ppi_bidirectionality(ppi_data, interaction_columns, verbose=verbose)
    simplified_ppi = simplify_ppi(unidirectional_ppi, interaction_columns, score, verbose=verbose)
    if rnaseq_genes is not None:
        if complex_sep is None:
            simplified_ppi = filter_ppi_by_proteins(ppi_data=simplified_ppi,
                                                    proteins=rnaseq_genes,
                                                    upper_letter_comparison=upper_letter_comparison,
                                                    )
        else:
            simplified_ppi = filter_complex_ppi_by_proteins(ppi_data=simplified_ppi,
                                                            proteins=rnaseq_genes,
                                                            complex_sep=complex_sep,
                                                            interaction_columns=('A', 'B'),
                                                            upper_letter_comparison=upper_letter_comparison
                                                            )
    simplified_ppi = simplified_ppi.drop_duplicates().reset_index(drop=True)
    return simplified_ppi


### Manipulate PPI networks
def remove_ppi_bidirectionality(ppi_data, interaction_columns, verbose=True):
    '''
    This function removes duplicate interactions. For example, when P1-P2 and P2-P1 interactions are present in the
    dataset, only one of them will remain.
    '''
    if verbose:
        print('Removing bidirectionality of PPI network')
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]
    IA = ppi_data[[header_interactorA, header_interactorB]]
    IB = ppi_data[[header_interactorB, header_interactorA]]
    IB.columns = [header_interactorA, header_interactorB]
    repeated_interactions = pd.merge(IA, IB, on=[header_interactorA, header_interactorB])
    repeated = list(np.unique(repeated_interactions.values.flatten()))
    df =  pd.DataFrame(combinations(sorted(repeated), 2), columns=[header_interactorA, header_interactorB])
    df = df[[header_interactorB, header_interactorA]]   # To keep lexicographically sorted interactions
    df.columns = [header_interactorA, header_interactorB]
    unidirectional_ppi = pd.merge(ppi_data, df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    unidirectional_ppi.reset_index(drop=True, inplace=True)
    return unidirectional_ppi


def simplify_ppi(ppi_data, interaction_columns, score=None, verbose=True):
    '''
    This functions reduces the PPI dataframe into a simpler version with only three columns (A, B and score).
    '''
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


def filter_ppi_by_proteins(ppi_data, proteins, complex_sep=None, upper_letter_comparison=True, interaction_columns=('A', 'B')):
    col_a = interaction_columns[0]
    col_b = interaction_columns[1]

    integrated_ppi = ppi_data.copy()

    if upper_letter_comparison:
        integrated_ppi[col_a] = integrated_ppi[col_a].apply(lambda x: str(x).upper())
        integrated_ppi[col_b] = integrated_ppi[col_b].apply(lambda x: str(x).upper())
        prots = set([str(p).upper() for p in proteins])
    else:
        prots = set(proteins)

    if complex_sep is not None:
        integrated_ppi = filter_complex_ppi_by_proteins(ppi_data=integrated_ppi,
                                                        proteins=prots,
                                                        complex_sep=complex_sep,
                                                        upper_letter_comparison=False, # Because it was ran above
                                                        interaction_columns=interaction_columns,
                                                        )
    else:
        integrated_ppi = integrated_ppi[(integrated_ppi[col_a].isin(prots)) & (integrated_ppi[col_b].isin(prots))]
    integrated_ppi = integrated_ppi.reset_index(drop=True)
    return integrated_ppi


def bidirectional_ppi_for_cci(ppi_data, interaction_columns=('A', 'B'), verbose=True):
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


def filter_ppi_network(ppi_data, contact_proteins, mediator_proteins=None, reference_list=None, bidirectional=True,
                       interaction_type='contacts', interaction_columns=('A', 'B'), verbose=True):
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

    if bidirectional:
        new_ppi_data = bidirectional_ppi_for_cci(ppi_data=new_ppi_data,
                                                 interaction_columns=interaction_columns,
                                                 verbose=verbose)
    return new_ppi_data


def get_genes_from_complexes(ppi_data, complex_sep='&', interaction_columns=('A', 'B')):
    col_a = interaction_columns[0]
    col_b = interaction_columns[1]

    col_a_genes = set()
    col_b_genes = set()

    complexes = dict()
    complex_a = set()
    complex_b = set()
    for idx, row in ppi_data.iterrows():
        prot_a = row[col_a]
        prot_b = row[col_b]

        if complex_sep in prot_a:
            comp = set([l for l in prot_a.split(complex_sep)])
            complexes[prot_a] = comp
            complex_a = complex_a.union(comp)
        else:
            col_a_genes.add(prot_a)

        if complex_sep in prot_b:
            comp = set([r for r in prot_b.split(complex_sep)])
            complexes[prot_b] = comp
            complex_b = complex_b.union(comp)
        else:
            col_b_genes.add(prot_b)

    return col_a_genes, complex_a, col_b_genes, complex_b, complexes


def filter_complex_ppi_by_proteins(ppi_data, proteins, complex_sep='&', upper_letter_comparison=True,
                                   interaction_columns=('A', 'B')):
    col_a = interaction_columns[0]
    col_b = interaction_columns[1]

    integrated_ppi = ppi_data.copy()

    if upper_letter_comparison:
        integrated_ppi[col_a] = integrated_ppi[col_a].apply(lambda x: str(x).upper())
        integrated_ppi[col_b] = integrated_ppi[col_b].apply(lambda x: str(x).upper())
        prots = set([str(p).upper() for p in proteins])
    else:
        prots = set(proteins)

    col_a_genes, complex_a, col_b_genes, complex_b, complexes = get_genes_from_complexes(ppi_data=integrated_ppi,
                                                                                         complex_sep=complex_sep,
                                                                                         interaction_columns=interaction_columns
                                                                                         )

    shared_a_genes = set(col_a_genes & prots)
    shared_b_genes = set(col_b_genes & prots)

    shared_a_complexes = set(complex_a & prots)
    shared_b_complexes = set(complex_b & prots)

    integrated_a_complexes = set()
    integrated_b_complexes = set()
    for k, v in complexes.items():
        if all(p in shared_a_complexes for p in v):
            integrated_a_complexes.add(k)
        elif all(p in shared_b_complexes for p in v):
            integrated_b_complexes.add(k)

    integrated_a = shared_a_genes.union(integrated_a_complexes)
    integrated_b = shared_b_genes.union(integrated_b_complexes)

    filter = (integrated_ppi[col_a].isin(integrated_a)) & (integrated_ppi[col_b].isin(integrated_b))
    integrated_ppi = ppi_data.loc[filter].reset_index(drop=True)

    return integrated_ppi


### These functions below are useful for filtering PPI networks based on GO terms
def get_filtered_ppi_network(ppi_data, contact_proteins, mediator_proteins=None, reference_list=None,
                             interaction_type='contacts', interaction_columns=('A', 'B'), verbose=True):
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


def get_all_to_all_ppi(ppi_data, proteins, interaction_columns=('A', 'B')):
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]
    new_ppi_data = ppi_data.loc[ppi_data[header_interactorA].isin(proteins) & ppi_data[header_interactorB].isin(proteins)]
    new_ppi_data = new_ppi_data.drop_duplicates()
    return new_ppi_data


def get_one_group_to_other_ppi(ppi_data, proteins_a, proteins_b, interaction_columns=('A', 'B')):
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]
    direction1 = ppi_data.loc[ppi_data[header_interactorA].isin(proteins_a) & ppi_data[header_interactorB].isin(proteins_b)]
    direction2 = ppi_data.loc[ppi_data[header_interactorA].isin(proteins_b) & ppi_data[header_interactorB].isin(proteins_a)]
    direction2.columns = [header_interactorB, header_interactorA, 'score']
    direction2 = direction2[[header_interactorA, header_interactorB, 'score']]
    new_ppi_data = pd.concat([direction1, direction2], ignore_index=True).drop_duplicates()
    return new_ppi_data