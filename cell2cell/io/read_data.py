# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pickle
import os
import pandas as pd
import numpy as np
from cell2cell.preprocessing import rnaseq, ppi
from cell2cell.io.directories import get_files_from_directory


def load_table(filename, format='auto', sep='\t', sheet_name=0, compression=None, verbose=True, **kwargs):
    '''Opens a file containing a table into a pandas dataframe.

    Parameters
    ----------
    filename : str
        Absolute path to a file storing a table.

    format : str, default='auto'
        Format of the file.
        Options are:

        - 'auto' : Automatically determines the format given
            the file extension. Files ending with .gz will be
            consider as tsv files.
        - 'excel' : An excel file, either .xls or .xlsx
        - 'csv' : Comma separated value format
        - 'tsv' : Tab separated value format
        - 'txt' : Text file

    sep : str, default='\t'
        Separation between columns. Examples are: '\t', ' ', ';', ',', etc.

    sheet_name : str, int, list, or None, default=0
        Strings are used for sheet names. Integers are used in zero-indexed
        sheet positions. Lists of strings/integers are used to request
        multiple sheets. Specify None to get all sheets.
        Available cases:

        - Defaults to 0: 1st sheet as a DataFrame
        - 1: 2nd sheet as a DataFrame
        - "Sheet1": Load sheet with name “Sheet1”
        - [0, 1, "Sheet5"]: Load first, second and sheet named
            “Sheet5” as a dict of DataFrame
        - None: All sheets.

    compression : str, or None, default=‘infer’
        For on-the-fly decompression of on-disk data. If ‘infer’, detects
        compression from the following extensions: ‘.gz’, ‘.bz2’, ‘.zip’, or ‘.xz’
        (otherwise no decompression). If using ‘zip’, the ZIP file must contain
        only one data file to be read in. Set to None for no decompression.
        Options: {‘infer’, ‘gzip’, ‘bz2’, ‘zip’, ‘xz’, None}

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    **kwargs : dict
        Extra arguments for loading files with the respective pandas function
        given the format of the file.

    Returns
    -------
    table : pandas.DataFrame
        Dataframe containing the table stored in a file.
    '''
    if filename is None:
        return None

    if format == 'auto':
        if ('.gz' in filename):
            format = 'csv'
            sep = '\t'
            compression='gzip'
        elif ('.xlsx' in filename) or ('.xls' in filename):
            format = 'excel'
        elif ('.csv' in filename):
            format = 'csv'
            sep = ','
            compression=None
        elif ('.tsv' in filename) or ('.txt' in filename):
            format = 'csv'
            sep = '\t'
            compression=None

    if format == 'excel':
        table = pd.read_excel(filename, sheet_name=sheet_name, **kwargs)
    elif (format == 'csv') | (format == 'tsv') | (format == 'txt'):
        table = pd.read_csv(filename, sep=sep, compression=compression, **kwargs)
    else:
        if verbose:
            print("Specify a correct format")
        return None
    if verbose:
        print(filename + ' was correctly loaded')
    return table


def load_tables_from_directory(pathname, extension, sep='\t', sheet_name=0, compression=None, verbose=True, **kwargs):
    '''Opens all tables with the same extension in a folder.

    Parameters
    ----------
    pathname : str
        Full path of the folder to explore.

    extension : str
        Extension of the file.
        Options are:

        - 'excel' : An excel file, either .xls or .xlsx
        - 'csv' : Comma separated value format
        - 'tsv' : Tab separated value format
        - 'txt' : Text file

    sep : str, default='\t'
        Separation between columns. Examples are: '\t', ' ', ';', ',', etc.

    sheet_name : str, int, list, or None, default=0
        Strings are used for sheet names. Integers are used in zero-indexed
        sheet positions. Lists of strings/integers are used to request
        multiple sheets. Specify None to get all sheets.
        Available cases:

        - Defaults to 0: 1st sheet as a DataFrame
        - 1: 2nd sheet as a DataFrame
        - "Sheet1": Load sheet with name “Sheet1”
        - [0, 1, "Sheet5"]: Load first, second and sheet named
            “Sheet5” as a dict of DataFrame
        - None: All sheets.

    compression : str, or None, default=‘infer’
        For on-the-fly decompression of on-disk data. If ‘infer’, detects
        compression from the following extensions: ‘.gz’, ‘.bz2’, ‘.zip’, or ‘.xz’
        (otherwise no decompression). If using ‘zip’, the ZIP file must contain
        only one data file to be read in. Set to None for no decompression.
        Options: {‘gzip’, ‘bz2’, ‘zip’, ‘xz’, None}

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    **kwargs : dict
        Extra arguments for loading files with the respective pandas function
        given the format of the file.

    Returns
    -------
    data : dict
        Dictionary containing the tables (pandas.DataFrame) loaded from the files.
        Keys are the filenames without the extension and values are the dataframes.
    '''
    assert extension in ['excel', 'csv', 'tsv', 'txt'], "Enter a valid `extension`."

    filenames = get_files_from_directory(pathname=pathname,
                                         dir_in_filepath=True)

    data = dict()
    if compression is None:
        comp = ''
    else:
        assert compression in ['gzip', 'bz2', 'zip', 'xz'], "Enter a valid `compression`."
        comp = '.' + compression
    for filename in filenames:
        if filename.endswith('.' + extension + comp):
            print('Loading {}'.format(filename))
            basename = os.path.basename(filename)
            sample = basename.split('.' + extension)[0]
            data[sample] = load_table(filename=filename,
                                      format=extension,
                                      sep=sep,
                                      sheet_name=sheet_name,
                                      compression=compression,
                                      verbose=verbose, **kwargs)
    return data


def load_rnaseq(rnaseq_file, gene_column, drop_nangenes=True, log_transformation=False, verbose=True, **kwargs):
    '''
    Loads a gene expression matrix for a RNA-seq experiment. Preprocessing
    steps can be done on-the-fly.

    Parameters
    ----------
    rnaseq_file : str
        Absolute path to a file containing a gene expression matrix. Genes
        are rows and cell-types/tissues/samples are columns.

    gene_column : str
        Column name where the gene labels are contained.

    drop_nangenes : boolean, default=True
        Whether dropping empty genes across all columns.

    log_transformation : boolean, default=False
        Whether applying a log10 transformation on the data.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    **kwargs : dict
        Extra arguments for loading files the function
        cell2cell.io.read_data.load_table

    Returns
    -------
    rnaseq_data : pandas.DataFrame
        Gene expression data for a bulk RNA-seq experiment or a single-cell
        experiment after aggregation into cell types. Columns are
        cell-types/tissues/samples and rows are genes.
    '''
    if verbose:
        print("Opening RNAseq datasets from {}".format(rnaseq_file))
    rnaseq_data = load_table(rnaseq_file, verbose=verbose, **kwargs)
    if gene_column is not None:
        rnaseq_data = rnaseq_data.set_index(gene_column)
    # Keep only numeric datasets
    rnaseq_data = rnaseq_data.select_dtypes([np.number])

    if drop_nangenes:
        rnaseq_data = rnaseq.drop_empty_genes(rnaseq_data)

    if log_transformation:
        rnaseq_data = rnaseq.log10_transformation(rnaseq_data)

    rnaseq_data = rnaseq_data.drop_duplicates()
    return rnaseq_data


def load_metadata(metadata_file, cell_labels=None, index_col=None, **kwargs):
    '''Loads a metadata table for a given list of cells.

    Parameters
    ----------
    metadata_file : str
        Absolute path to a file containing a metadata table for
        cell-types/tissues/samples in a RNA-seq dataset.

    cell_labels : list, default=None
        List of cell-types/tissues/samples to consider. Names must
        match the labels in the metadata table. These names must
        be contained in the values of the column indicated
        by index_col.

    index_col : str, default=None
        Column to be consider the index of the metadata.
        If None, the index will be the numbers of the rows.

    **kwargs : dict
        Extra arguments for loading files the function
        cell2cell.io.read_data.load_table

    Returns
    -------
    meta : pandas.DataFrame
        Metadata for the cell-types/tissues/samples provided.
    '''
    meta = load_table(metadata_file, **kwargs)
    if index_col is None:
        index_col = list(meta.columns)[0]
        indexing = False
    else:
        indexing = True
    if cell_labels is not None:
        meta = meta.loc[meta[index_col].isin(cell_labels)]

    if indexing:
        meta.set_index(index_col, inplace=True)
    return meta


def load_cutoffs(cutoff_file, gene_column=None, drop_nangenes=True, log_transformation=False, verbose=True, **kwargs):
    '''Loads a table of cutoff of thresholding values for each gene.

    Parameters
    ----------
    cutoff_file : str
        Absolute path to a file containing thresholding values for genes.
        Genes are rows and threshold values are in the only column beyond
        the one containing the gene names.

    gene_column : str, default=None
        Column name where the gene labels are contained. If None, the
        first column will be assummed to contain gene names.

    drop_nangenes : boolean, default=True
        Whether dropping empty genes across all columns.

    log_transformation : boolean, default=False
        Whether applying a log10 transformation on the data.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    **kwargs : dict
        Extra arguments for loading files the function
        cell2cell.io.read_data.load_table

    Returns
    -------
    cutoff_data : pandas.DataFrame
        Dataframe with the cutoff values for each gene. Rows are genes
        and just one column is included, which corresponds to 'value',
        wherein the thresholding or cutoff values are contained.
    '''
    if verbose:
        print("Opening Cutoff datasets from {}".format(cutoff_file))
    cutoff_data = load_table(cutoff_file, verbose=verbose, **kwargs)
    if gene_column is not None:
        cutoff_data = cutoff_data.set_index(gene_column)
    else:
        cutoff_data = cutoff_data.set_index(cutoff_data.columns[0])

    # Keep only numeric datasets
    cutoff_data = cutoff_data.select_dtypes([np.number])

    if drop_nangenes:
        cutoff_data = rnaseq.drop_empty_genes(cutoff_data)

    if log_transformation:
        cutoff_data = rnaseq.log10_transformation(cutoff_data)

    cols = list(cutoff_data.columns)
    cols[0] = 'value'
    cutoff_data.columns = cols
    return cutoff_data


def load_ppi(ppi_file, interaction_columns, sort_values=None, score=None, rnaseq_genes=None, complex_sep=None,
             dropna=False, strna='', upper_letter_comparison=False, verbose=True, **kwargs):
    '''Loads a list of protein-protein interactions from a table and
    returns it in a simplified format.

    Parameters
    ----------
    ppi_file : str
        Absolute path to a file containing a list of protein-protein
        interactions.

    interaction_columns : tuple
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors. Example: ('partner_A', 'partner_B').

    sort_values : str, default=None
        Column name of a column used for sorting the table. If it is not None,
        that the column, and the whole dataframe, we will be ordered in an
        ascending manner.

    score : str, default=None
        Column name of a column containing weights to consider in the cell-cell
        interactions/communication analyses. If None, no weights are used and
        PPIs are assumed to have an equal contribution to CCI and CCC scores.

    rnaseq_genes : list, default=None
        List of genes in a RNA-seq dataset to filter the list of PPIs. If None,
        the entire list will be used.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44". If None, it is assummed
        that the list does not contains complexes.

    dropna : boolean, default=False
        Whether dropping PPIs with any missing information.

    strna : str, default=''
        If dropna is False, missing values will be filled with strna.

    upper_letter_comparison : boolean, default=False
        Whether making uppercase the gene names in the expression matrices and the
        protein names in the ppi_data to match their names and integrate their
        respective expression level. Useful when there are inconsistencies in the
        names between the expression matrix and the ligand-receptor annotations.

    **kwargs : dict
        Extra arguments for loading files the function
        cell2cell.io.read_data.load_table

    Returns
    -------
    simplified_ppi : pandas.DataFrame
        A simplified list of PPIs. In this case, interaction_columns are renamed
        into 'A' and 'B' for the first and second interacting proteins, respectively.
        A third column 'score' is included, containing weights of PPIs.
    '''
    if verbose:
        print("Opening PPI datasets from {}".format(ppi_file))
    ppi_data = load_table(ppi_file, verbose=verbose,  **kwargs)

    simplified_ppi = ppi.preprocess_ppi_data(ppi_data=ppi_data,
                                             interaction_columns=interaction_columns,
                                             sort_values=sort_values,
                                             score=score,
                                             rnaseq_genes=rnaseq_genes,
                                             complex_sep=complex_sep,
                                             dropna=dropna,
                                             strna=strna,
                                             upper_letter_comparison=upper_letter_comparison,
                                             verbose=verbose)
    return simplified_ppi


def load_go_terms(go_terms_file, verbose=True):
    '''Loads GO term information from a obo-basic file.

    Parameters
    ----------
    go_terms_file : str
        Absolute path to an obo file. It could be an URL as for example:
        go_terms_file = 'http://purl.obolibrary.org/obo/go/go-basic.obo'

    verbose : boolean, default=True
            Whether printing or not steps of the analysis.

    Returns
    -------
    go_terms : networkx.Graph
        NetworkX Graph containing GO terms datasets from .obo file.
    '''
    import cell2cell.external.goenrich as goenrich

    if verbose:
        print("Opening GO terms from {}".format(go_terms_file))
    go_terms = goenrich.ontology(go_terms_file)
    if verbose:
        print(go_terms_file + ' was correctly loaded')
    return go_terms


def load_go_annotations(goa_file, experimental_evidence=True, verbose=True):
    '''Loads GO annotations for each gene in a given organism.

    Parameters
    ----------
    goa_file : str
        Absolute path to an ga file. It could be an URL as for example:
        goa_file = 'http://current.geneontology.org/annotations/wb.gaf.gz'

    experimental_evidence : boolean, default=True
        Whether considering only annotations with experimental evidence
        (at least one article/evidence).

    verbose : boolean, default=True
            Whether printing or not steps of the analysis.

    Returns
    -------
    goa : pandas.DataFrame
        Dataframe containing information about GO term annotations of each
        gene for a given organism according to the ga file.
    '''
    import cell2cell.external.goenrich as goenrich

    if verbose:
        print("Opening GO annotations from {}".format(goa_file))

    goa = goenrich.goa(goa_file, experimental_evidence)
    goa_cols = list(goa.columns)
    goa = goa[goa_cols[:3] + [goa_cols[4]]]
    new_cols = ['db', 'Gene', 'Name', 'GO']
    goa.columns = new_cols
    if verbose:
        print(goa_file + ' was correctly loaded')
    return goa


def load_variable_with_pickle(filename):
    '''Imports a large size variable stored in a file previously
    exported with pickle.

    Parameters
    ----------
    filename : str
        Absolute path to a file storing a python variable that
        was previously created by using pickle.

    Returns
    -------
    variable : a python variable
        The variable of interest.
    '''

    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(filename)
    with open(filename, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    variable = pickle.loads(bytes_in)
    return variable


def load_tensor(filename, backend=None, device=None):
    '''Imports a communication tensor that could be used
    with Tensor-cell2cell.

    Parameters
    ----------
    filename : str
        Absolute path to a file storing a communication tensor
        that was previously saved by using pickle.

    backend : str, default=None
        Backend that TensorLy will use to perform calculations
        on this tensor. When None, the default backend used is
        the currently active backend, usually is ('numpy'). Options are:
        {'cupy', 'jax', 'mxnet', 'numpy', 'pytorch', 'tensorflow'}

    device : str, default=None
        Device to use when backend allows using multiple devices. Options are:
         {'cpu', 'cuda:0', None}

    Returns
    -------
    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor generated with any of the tensor class in
        cell2cell.tensor.
    '''
    interaction_tensor = load_variable_with_pickle(filename)

    if 'tl' not in globals():
        import tensorly as tl

    if backend is not None:
        tl.set_backend(backend)

    if device is None:
        interaction_tensor.tensor = tl.tensor(interaction_tensor.tensor)
        interaction_tensor.loc_nans = tl.tensor(interaction_tensor.loc_nans)
        interaction_tensor.loc_zeros = tl.tensor(interaction_tensor.loc_zeros)
        if interaction_tensor.mask is not None:
            interaction_tensor.mask = tl.tensor(interaction_tensor.mask)
    else:
        if tl.get_backend() in ['pytorch', 'tensorflow']:  # Potential TODO: Include other backends that support different devices
            interaction_tensor.tensor = tl.tensor(interaction_tensor.tensor, device=device)
            interaction_tensor.loc_nans = tl.tensor(interaction_tensor.loc_nans, device=device)
            interaction_tensor.loc_zeros = tl.tensor(interaction_tensor.loc_zeros, device=device)
            if interaction_tensor.mask is not None:
                interaction_tensor.mask = tl.tensor(interaction_tensor.mask, device=device)
        else:
            interaction_tensor.tensor = tl.tensor(interaction_tensor.tensor)
            interaction_tensor.loc_nans = tl.tensor(interaction_tensor.loc_nans)
            interaction_tensor.loc_zeros = tl.tensor(interaction_tensor.loc_zeros)
            if interaction_tensor.mask is not None:
                interaction_tensor.mask = tl.tensor(interaction_tensor.mask)
    return interaction_tensor


def load_tensor_factors(filename):
    '''Imports factors previously exported from a tensor
    decomposition done in a cell2cell.tensor.BaseTensor-like object.

    Parameters
    ----------
    filename : str
        Absolute path to a file storing an excel file containing
        the factors, their loadings, and element names for each
        of the dimensions of a previously decomposed tensor.

    Returns
    -------
    factors : collections.OrderedDict
        An ordered dictionary wherein keys are the names of each
        tensor dimension, and values are the loadings in a pandas.DataFrame.
        In this dataframe, rows are the elements of the respective dimension
        and columns are the factors from the tensor factorization. Values
        are the corresponding loadings.
    '''
    from collections import OrderedDict

    xls = pd.ExcelFile(filename)

    factors = OrderedDict()
    for sheet_name in xls.sheet_names:
        factors[sheet_name] = xls.parse(sheet_name, index_col=0)

    return factors