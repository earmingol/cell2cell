# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd

class Cell:
    '''Specific cell/tissue/organ element containing associated RNAseq data.

    Parameters
    ----------
    sc_rnaseq_data : pandas DataFrame object
        Object containing a column for the RNAseq values (usually TPM). Gene ids are the indexes of this DataFrame.

    verbose : boolean, True by default
        Value to indicate if stdout will be printed.

    Attributes
    ----------
    id : int
        Identifier number of the each instance generated.

    type : str
        Identifier name of the respective cell/tissue/organ that RNAseq data belongs to.

    rnaseq_data : pandas DataFrame object
        Copy of sc_rnaseq_data.

    binary_ppi : pandas DataFrame object
        Object containing a ppi table, but the values in each cell corresponds to the binary expression of the respective
        protein/gene in the original ppi table. Each protein name has been replaced by the respective expression value
        in sc_rnaseq_data.
    '''
    _id_counter = 0

    def __init__(self, sc_rnaseq_data, verbose=True):
        self.id = Cell._id_counter
        Cell._id_counter += 1

        self.type = str(sc_rnaseq_data.columns[-1])

        # RNAseq data
        self.rnaseq_data = sc_rnaseq_data.copy()
        self.rnaseq_data.columns = ['value']

        # Binary ppi data
        self.binary_ppi = pd.DataFrame(columns=['A', 'B', 'score'])

        # Object created
        if verbose:
            print("New cell instance created for " + self.type)

    def __del__(self):
        Cell._id_counter -= 1

    def __str__(self):
        return str(self.type)

    __repr__ = __str__


def cells_from_rnaseq(rnaseq_data, cell_columns = None, verbose=True):
    '''Create new instances of Cells based on the RNAseq data and the cell/tissue/organ types provided.

    Parameters
    ----------
    rnaseq_data : pandas DataFrame object
        Object containing a multiple columns for the RNAseq values (usually TPM), where each one represents a
        cell/tissue/organ type. Gene ids are the indexes of this DataFrame.

    cell columns : array-like
        List of strings containing the names of cell types in RNAseq data.

    verbose : boolean, True by default
        Value to indicate if stdout will be printed.

    Returns
    -------
    cells : dict
        Dictionary containing all Cell instances generated from RNAseq data to represent each cell/tissue/organ type.
        The keys of this dictionary are the names of the corresponding Cell instance.
    '''
    if verbose:
        print("Generating objects according to RNAseq data provided")
    cells = dict()
    if cell_columns is None:
        cell_columns = rnaseq_data.columns

    for cell in cell_columns:
        cells[cell] = Cell(rnaseq_data[[cell]], verbose=verbose)
    return cells
