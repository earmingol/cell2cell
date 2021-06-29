# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd

class Cell:
    '''Specific cell-type/tissue/organ element in a RNAseq dataset.

    Parameters
    ----------
    sc_rnaseq_data : pandas.DataFrame
        A gene expression matrix. Contains only one column that
        corresponds to cell-type/tissue/sample, while the genes
        are rows and the specific. Column name will be the label
        of the instance.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Attributes
    ----------
    id : int
        ID number of the instance generated.

    type : str
        Name of the respective cell-type/tissue/sample.

    rnaseq_data : pandas.DataFrame
        Copy of sc_rnaseq_data.

    weighted_ppi : pandas.DataFrame
        Dataframe created from a list of protein-protein interactions,
        here the columns of the interacting proteins are replaced by
        a score or a preprocessed gene expression of the respective
        proteins.
    '''
    _id_counter = 0  # Number of active instances
    _id = 0 # Unique ID

    def __init__(self, sc_rnaseq_data, verbose=True):
        self.id = Cell._id
        Cell._id_counter += 1
        Cell._id += 1

        self.type = str(sc_rnaseq_data.columns[-1])

        # RNAseq datasets
        self.rnaseq_data = sc_rnaseq_data.copy()
        self.rnaseq_data.columns = ['value']

        # Binary ppi datasets
        self.weighted_ppi = pd.DataFrame(columns=['A', 'B', 'score'])

        # Object created
        if verbose:
            print("New cell instance created for " + self.type)

    def __del__(self):
        Cell._id_counter -= 1

    def __str__(self):
        return str(self.type)

    __repr__ = __str__


def get_cells_from_rnaseq(rnaseq_data, cell_columns=None, verbose=True):
    '''
    Creates new instances of Cell based on the RNAseq data of each
    cell-type/tissue/sample in a gene expression matrix.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    cell_columns : array-like, default=None
        List of names of cell-types/tissues/samples in the dataset
        to be used. If None, all columns will be used.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    cells : dict
        Dictionary containing all Cell instances generated from a RNAseq dataset.
        The keys of this dictionary are the names of the corresponding Cell instances.
    '''
    if verbose:
        print("Generating objects according to RNAseq datasets provided")
    cells = dict()
    if cell_columns is None:
        cell_columns = rnaseq_data.columns

    for cell in cell_columns:
        cells[cell] = Cell(rnaseq_data[[cell]], verbose=verbose)
    return cells
