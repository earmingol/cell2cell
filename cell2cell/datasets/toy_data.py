import numpy as np
import pandas as pd


def generate_toy_rnaseq():
    '''Generates a toy RNA-seq dataset

    Returns
    -------
    rnaseq : pandas.DataFrame
        DataFrame contianing the toy RNA-seq dataset. Columns
        are cells and rows are genes.
    '''
    data = np.asarray([[5, 10, 8, 15, 2],
                       [15, 5, 20, 1, 30],
                       [18, 12, 5, 40, 20],
                       [9, 30, 22, 5, 2],
                       [2, 1, 1, 27, 15],
                       [30, 11, 16, 5, 12],
                       ])

    rnaseq = pd.DataFrame(data,
                          index=['Protein-A', 'Protein-B', 'Protein-C', 'Protein-D', 'Protein-E', 'Protein-F'],
                          columns=['C1', 'C2', 'C3', 'C4', 'C5']
                          )
    rnaseq.index.name = 'gene_id'
    return rnaseq


def generate_toy_ppi(prot_complex=False):
    '''Generates a toy list of protein-protein interactions.

    Parameters
    ----------
    prot_complex : boolean, default=False
        Whether including PPIs where interactors could contain
        multimeric complexes.

    Returns
    -------
    ppi : pandas.DataFrame
        Dataframe containing PPIs. Columns are 'A' (first interacting
        partners), 'B' (second interacting partners) and 'score'
        for weighting each PPI.
    '''
    if prot_complex:
        data = np.asarray([['Protein-A', 'Protein-B'],
                           ['Protein-B', 'Protein-C'],
                           ['Protein-C', 'Protein-A'],
                           ['Protein-B', 'Protein-B'],
                           ['Protein-B', 'Protein-A'],
                           ['Protein-E', 'Protein-F'],
                           ['Protein-F', 'Protein-F'],
                           ['Protein-C&Protein-E', 'Protein-F'],
                           ['Protein-B', 'Protein-E'],
                           ['Protein-A&Protein-B', 'Protein-F'],
                           ])
    else:
        data = np.asarray([['Protein-A', 'Protein-B'],
                           ['Protein-B', 'Protein-C'],
                           ['Protein-C', 'Protein-A'],
                           ['Protein-B', 'Protein-B'],
                           ['Protein-B', 'Protein-A'],
                           ['Protein-E', 'Protein-F'],
                           ['Protein-F', 'Protein-F'],
                           ['Protein-C', 'Protein-F'],
                           ['Protein-B', 'Protein-E'],
                           ['Protein-A', 'Protein-F'],
                           ])
    ppi = pd.DataFrame(data, columns=['A', 'B'])
    ppi = ppi.assign(score=1.0)
    return ppi


def generate_toy_metadata():
    '''Generates metadata for cells in the toy RNA-seq dataset.

    Returns
    -------
    metadata : pandas.DataFrame
        DataFrame with metadata for each cell. Metadata contains the
        major groups of those cells.
    '''
    data = np.asarray([['C1', 'G1'],
                       ['C2', 'G2'],
                       ['C3', 'G3'],
                       ['C4', 'G3'],
                       ['C5', 'G1']
                       ])

    metadata = pd.DataFrame(data, columns=['#SampleID', 'Groups'])
    return metadata


def generate_toy_distance():
    '''Generates a square matrix with cell-cell distance.

    Returns
    -------
    distance : pandas.DataFrame
        DataFrame with Euclidean-like distance between each
        pair of cells in the toy RNA-seq dataset.
    '''
    data = np.asarray([[0.0, 10.0, 12.0, 5.0, 3.0],
                       [10.0, 0.0, 15.0, 8.0, 9.0],
                       [12.0, 15.0, 0.0, 4.5, 7.5],
                       [5.0, 8.0, 4.5, 0.0, 6.5],
                       [3.0, 9.0, 7.5, 6.5, 0.0],
                       ])
    distance = pd.DataFrame(data,
                            index=['C1', 'C2', 'C3', 'C4', 'C5'],
                            columns=['C1', 'C2', 'C3', 'C4', 'C5']
                            )
    return distance