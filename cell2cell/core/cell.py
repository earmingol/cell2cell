# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd
from cell2cell.io.read_data import load_rnaseq

class Cell:
    '''Specific cell/tissue/organ element containing associated RNAseq data.

    Parameters
    ----------
    RNAseq_data : pandas DataFrame object
        Object containing a column for gene ids as well as another column for the RNAseq valuues (TPM).

    gene_column : str
        Name of the column containing gene ids.

    cell_column : str
        Name of the column containing cell/tissue/organ expression data.


    Attributes
    ----------
    id : int
        Identifier number of the each instance generated.

    type : str
        Identifier name of the respective cell/tissue/organ that RNAseq data belongs to.

    rnaseq_data : pandas DataFrame object
        Object containing a column for gene ids as well as another column for the RNAseq valuues (TPM).
    '''
    _id_counter = 0

    def __init__(self, sc_rnaseq_data):
        self.id = Cell._id_counter
        Cell._id_counter += 1

        self.type = str(sc_rnaseq_data.columns[-1])

        # RNAseq data
        self.rnaseq_data = sc_rnaseq_data.copy()
        self.rnaseq_data.columns = ['value']

        # Binary ppi data
        self.binary_ppi = pd.DataFrame(columns=['A', 'B'])

        # Object created
        print("New cell instance created for " + self.type)

    def __del__(self):
        Cell._id_counter -= 1

    def __str__(self):
        return str(self.type)

    __repr__ = __str__


def cells_from_rnaseq(rnaseq_file, gene_column, **kwargs):
    '''Create new instances of Cells based on the RNAseq data and the cell/tissue/organ types provided.

    Parameters
    ----------
    rnaseq_file : str
        Path to the rnaseq data. This will be used to load an object containing a column for gene ids as well as columns
         for each cell/tissue/organ containing their respective expression data for each gene (TPM unit).

    cell columns : array-like, list
        List of strings containing the names of cell types in RNAseq data.

    gene_column : str
        Name of the column containing gene ids.

    **kwargs : keyworded arguments
        Arguments of function load_table.

    Returns
    -------
    cells : dict
        Dictionary containing all Cell instances generated from RNAseq data to represent each cell/tissue/organ type.
        The keys of this dictionary are the ids for the corresponding Cell instance.
    '''

    print("Generating objects according to RNAseq data provided")
    rnaseq_data = load_rnaseq(rnaseq_file, gene_column, kwargs)
    cells = {}
    for cell in rnaseq_data.columns:
        cells[Cell._id_counter - 1] = Cell(rnaseq_data[[cell]])
    return cells
