# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd

from sklearn.utils import resample

from cell2cell.preprocessing import rnaseq, ppi


def generate_random_rnaseq(size, row_names, random_state=None, verbose=True):
    '''
    Generates a RNA-seq dataset that is normally distributed gene-wise and size
    normalized (each column sums up to a million).

    Parameters
    ----------
    size : int
        Number of cell-types/tissues/samples (columns).

    row_names : array-like
        List containing the name of genes (rows).

    random_state : int, default=None
        Seed for randomization.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing gene expression given the list
        of genes for each cell-type/tissue/sample.
    '''
    if verbose:
        print('Generating random RNA-seq dataset.')
    columns = ['Cell-{}'.format(c) for c in range(1, size+1)]

    if random_state is not None:
        np.random.seed(random_state)
    data = np.random.randn(len(row_names), len(columns))    # Normal distribution
    min = np.abs(np.amin(data, axis=1))
    min = min.reshape((len(min), 1))

    data = data + min
    df = pd.DataFrame(data, index=row_names, columns=columns)
    if verbose:
        print('Normalizing random RNA-seq dataset (into TPM)')
    df = rnaseq.scale_expression_by_sum(df, axis=0, sum_value=1e6)
    return df


def generate_random_ppi(max_size, interactors_A, interactors_B=None, random_state=None, verbose=True):
    '''Generates a random list of protein-protein interactions.

    Parameters
    ----------
    max_size : int
        Maximum size of interactions to obtain. Since the PPIs
        are obtained by independently resampling interactors A and B
        rather than creating all possible combinations (it may demand too much
        memory), some PPIs can be duplicated and when dropping them
        results into a smaller number of PPIs than the max_size.

    interactors_A : list
        A list of protein names to include in the first column of
        the PPIs.

    interactors_B : list, default=None
        A list of protein names to include in the second columns
        of the PPIs. If None, interactors_A will be used as
        interactors_B too.

    random_state : int, default=None
        Seed for randomization.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    ppi_data : pandas.DataFrame
        DataFrame containing a list of protein-protein interactions.
        It has three columns: 'A', 'B', and 'score' for interactors
        A, B and weights of interactions, respectively.
    '''
    if interactors_B is not None:
        assert max_size <= len(interactors_A)*len(interactors_B), "The maximum size can't be greater than all combinations between partners A and B"
    else:
        assert max_size <= len(interactors_A)**2, "The maximum size can't be greater than all combinations of partners A"


    if verbose:
        print('Generating random PPI network.')

    def small_block_ppi(size, interactors_A, interactors_B, random_state):
        if random_state is not None:
            random_state += 1
        if interactors_B is None:
            interactors_B = interactors_A

        col_A = resample(interactors_A, n_samples=size, random_state=random_state)
        col_B = resample(interactors_B, n_samples=size, random_state=random_state)

        ppi_data = pd.DataFrame()
        ppi_data['A'] = col_A
        ppi_data['B'] = col_B
        ppi_data.assign(score=1.0)

        ppi_data = ppi.remove_ppi_bidirectionality(ppi_data, ('A', 'B'), verbose=verbose)
        ppi_data = ppi_data.drop_duplicates()
        ppi_data.reset_index(inplace=True, drop=True)
        return ppi_data

    ppi_data = small_block_ppi(max_size*2, interactors_A, interactors_B, random_state)

    # TODO: This part need to be fixed, it does not converge to the max_size -> len((set(A)) * len(set(B) - set(A)))
    # while ppi_data.shape[0] < size:
    #     if random_state is not None:
    #         random_state += 2
    #     b = small_block_ppi(size, interactors_A, interactors_B, random_state)
    #     print(b)
    #     ppi_data = pd.concat([ppi_data, b])
    #     ppi_data = ppi.remove_ppi_bidirectionality(ppi_data, ('A', 'B'), verbose=verbose)
    #     ppi_data = ppi_data.drop_duplicates()
    #     ppi_data.dropna()
    #     ppi_data.reset_index(inplace=True, drop=True)
    #     print(ppi_data.shape[0])

    if ppi_data.shape[0] > max_size:
        ppi_data = ppi_data.loc[list(range(max_size)), :]
    ppi_data.reset_index(inplace=True, drop=True)
    return ppi_data


def generate_random_cci_scores(cell_number, labels=None, symmetric=True, random_state=None):
    '''Generates a square cell-cell interaction
    matrix with random scores.

    Parameters
    ----------
    cell_number : int
        Number of cells.

    labels : list, default=None
        List containing labels for each cells. Length of
        this list must match the cell_number.

    symmetric : boolean, default=True
        Whether generating a symmetric CCI matrix.

    random_state : int, default=None
        Seed for randomization.

    Returns
    -------
    cci_matrix : pandas.DataFrame
        Matrix with rows and columns as cells. Values
        represent a random CCI score between 0 and 1.
    '''
    if labels is not None:
        assert len(labels) == cell_number, "Lenght of labels must match cell_number"
    else:
        labels = ['Cell-{}'.format(n) for n in range(1, cell_number+1)]

    if random_state is not None:
        np.random.seed(random_state)
    cci_scores = np.random.random((cell_number, cell_number))
    if symmetric:
        cci_scores = (cci_scores + cci_scores.T) / 2.
    cci_matrix = pd.DataFrame(cci_scores, index=labels, columns=labels)

    return cci_matrix


def generate_random_metadata(cell_labels, group_number):
    '''Randomly assigns groups to cell labels.

    Parameters
    ----------
    cell_labels : list
        A list of cell labels.

    group_number : int
        Number of major groups of cells.

    Returns
    -------
    metadata : pandas.DataFrame
        DataFrame containing the major groups that each cell
        received randomly (under column 'Group'). Cells are
        under the column 'Cell'.
    '''
    metadata = pd.DataFrame()
    metadata['Cell'] = cell_labels

    groups = list(range(1, group_number+1))
    metadata['Group'] = metadata['Cell'].apply(lambda x: np.random.choice(groups, 1)[0])
    return metadata
