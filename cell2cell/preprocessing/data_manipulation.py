import pandas as pd
import numpy as np


### Pre-process RNAseq data
def drop_empty_genes(rnaseq_data):
    data = rnaseq_data.copy()
    data = data.fillna(0)  # Put zeros to all missing values
    data = data.loc[data.sum(axis=1) != 0]  # Drop rows will 0 among all cell/tissues
    return data


def log10_transformation(rnaseq_data):
    ### Apply this only after applying "drop_empty_genes" function
    data = rnaseq_data.copy()
    data = data.apply(np.log10)
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


### Manipulate PPI networks
def remove_ppi_bidirectionality(PPI_data, header_interactorA, header_interactorB):
    print('Removing bidirectionality of PPI network')
    IA = PPI_data[[header_interactorA, header_interactorB]]
    IB = PPI_data[[header_interactorB, header_interactorA]]
    IB.columns = [header_interactorA, header_interactorB]
    repeated_interactions = pd.merge(IA, IB, on=[header_interactorA, header_interactorB])
    unidirectional_ppi = pd.merge(PPI_data, repeated_interactions, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    return unidirectional_ppi


def simplify_ppi(PPI_data, header_interactorA, header_interactorB, score = None):
    print('Simplying PPI network')
    if score is None:
        simple_PPI = PPI_data[[header_interactorA, header_interactorB]]
        simple_PPI = simple_PPI.assign(score = 1.0)
    else:
        simple_PPI = PPI_data[[header_interactorA, header_interactorB, score]]
    cols = ['A', 'B', 'score']
    simple_PPI.columns = cols
    return simple_PPI


def filter_ppi_network(PPI_data, header_interactorA, header_interactorB, product_filter, mediators = None, interaction_type = 'physical'):
    '''
    :param PPI_data:
    :param header_interactorA:
    :param header_interactorB:
    :param product_filter: List of genes/proteins consistent with IDs used in PPI
    :return:
    '''
    physical = PPI_data.loc[PPI_data[header_interactorA].isin(product_filter) & PPI_data[header_interactorB].isin(product_filter)]
    if interaction_type == 'physical':
        print('Filtering PPI network by using a product list')
        new_PPI_data = physical.drop_duplicates()
    else:
        mediated1 = PPI_data.loc[PPI_data[header_interactorA].isin(product_filter) & PPI_data[header_interactorB].isin(mediators)]
        mediated2 = PPI_data.loc[PPI_data[header_interactorA].isin(mediators) & PPI_data[header_interactorB].isin(product_filter)]
        mediated2.columns = [header_interactorB, header_interactorA, 'score']
        mediated2 = mediated2[[header_interactorA, header_interactorB, 'score']]
        mediated = pd.concat([mediated1, mediated2], ignore_index = True).drop_duplicates()
        if interaction_type == 'mediated':
            new_PPI_data = mediated
        elif interaction_type == 'combined':
            new_PPI_data = pd.concat([physical, mediated], ignore_index = True).drop_duplicates()
        else:
            raise NameError('Not valid interaction type to filter the PPI network')
    return new_PPI_data


### Gene Ontology related functions
def get_genes_from_go_terms(GO_data, go_filter, go_header = 'GO', gene_header = 'Gene'):
    print('Filtering genes by using GO terms')
    filtered_genes = list(np.unique(GO_data.loc[GO_data[go_header].isin(go_filter)][gene_header].values))
    return filtered_genes


def get_genes_from_parent_go_terms(GO_data, GO_terms, go_filter, go_header ='GO', gene_header ='Gene', verbose = False):
    parent_GOs = go_filter.copy()
    iter = len(parent_GOs)
    for i in range(iter):
        find_all_children_of_go_term(GO_terms, parent_GOs[i], parent_GOs, verbose = verbose)
    genes = get_genes_from_go_terms(GO_data, parent_GOs, go_header, gene_header)
    return genes


def find_all_children_of_go_term(graph, go_term, empty_list, verbose = True):
    import networkx
    for child in networkx.ancestors(graph, go_term):
        if child not in empty_list:
            if verbose:
                print('Retrieving children for ' + go_term)
            empty_list.append(child)
        find_all_children_of_go_term(graph, child, empty_list, verbose)