# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd

from itertools import combinations

from natsort import natsorted


### Preprocess a PPI table from a known list
def preprocess_ppi_data(ppi_data, interaction_columns, sort_values=None, score=None, rnaseq_genes=None, complex_sep=None,
             dropna=False, strna='', upper_letter_comparison=True, duplicates='highest', verbose=True):
    '''
    Preprocess a list of protein-protein interactions by
    removed bidirectionality and keeping the minimum number of columns.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    interaction_columns : tuple
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    sort_values : str, default=None
        Column name to sort PPIs in an ascending manner. If None,
        sorting is not done.

    rnaseq_genes : list, default=None
        List of protein or gene names to filter the PPIs.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    dropna : boolean, default=False
        Whether dropping incomplete PPIs (with NaN values).

    strna : str, default=''
        String to replace empty or NaN values with.

    upper_letter_comparison : boolean, default=True
        Whether making uppercase the gene names in the expression matrices and the
        protein names in the ppi_data to match their names and integrate their
        respective expression level. Useful when there are inconsistencies in the
        names between the expression matrix and the ligand-receptor annotations.

    duplicates : str, default='highest'
        What to do with a pair listed more than once, passed to
        `deduplicate_ppi_pairs` as its `keep`: 'highest' (the default) keeps the row
        with the largest score, 'lowest' the smallest and 'first' the one appearing
        first. Pass 'keep' to leave the repeated rows in the list, which is only
        meaningful when they carry different scores.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    simplified_ppi : pandas.DataFrame
        A simplified list of protein-protein interactions. It does not contains
        duplicated interactions in both directions (if ProtA-ProtB and
        ProtB-ProtA interactions are present, only the one that appears first
        is kept) either extra columns beyond interacting ones, and holds one row
        per interacting pair unless `duplicates='keep'`. It contains only
        three columns: 'A', 'B', 'score', wherein 'A' and 'B' are the interacting
        partners in the PPI and 'score' represents a weight of the interaction
        for computing cell-cell interactions/communication.
    '''
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
    if duplicates == 'keep':
        # Only rows that are identical throughout, so a pair listed twice with
        # different scores stays as two interactions
        simplified_ppi = simplified_ppi.drop_duplicates().reset_index(drop=True)
    else:
        simplified_ppi = deduplicate_ppi_pairs(simplified_ppi, interaction_columns=('A', 'B'),
                                               score='score', keep=duplicates,
                                               verbose=False)
    return simplified_ppi


### Manipulate PPI networks
def deduplicate_ppi_pairs(ppi_data, interaction_columns=('A', 'B'), score='score',
                          keep='highest', verbose=False):
    '''
    Keeps a single row for each interacting pair.

    A pair listed more than once is one interaction, so the rows have to be collapsed
    into one before the list is used: a repeated pair would otherwise contribute more
    than once to every cell-cell interaction score, and in the ligand-receptor search
    it would occupy several positions of the candidate set while being a single
    interaction.

    Which of the repeated rows to keep only matters when they carry different weights
    in the score column, and `keep` decides that.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs).

    interaction_columns : tuple, default=('A', 'B')
        Columns holding the two partners. Pairs are compared as given, so an
        interaction and its reverse are two pairs here -- use
        `remove_ppi_bidirectionality` for those.

    score : str, default='score'
        Column holding the weight of each interaction. Ignored if absent, in which
        case the repeated rows are indistinguishable and the first one is kept.

    keep : str, default='highest'
        Which row to keep among those listing the same pair:

        - 'highest' : the one with the largest score, the strongest evidence for the
          interaction when the score is a confidence or an expression weight.
        - 'lowest' : the one with the smallest score, the conservative choice.
        - 'first' : the one appearing first, whatever its score.

        Ties keep the row that appears first.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    Returns
    -------
    deduplicated_ppi : pandas.DataFrame
        The list with one row per pair, in the order the pairs first appear in
        `ppi_data`.

    Examples
    --------
    >>> import cell2cell as c2c
    >>> deduplicated = c2c.preprocessing.deduplicate_ppi_pairs(ppi_data)
    '''
    if keep not in ('highest', 'lowest', 'first'):
        raise ValueError("`keep` must be 'highest', 'lowest' or 'first'")
    header_interactorA, header_interactorB = interaction_columns

    scores = None
    if keep != 'first' and score in ppi_data.columns:
        scores = np.asarray(ppi_data[score].values, dtype=float)

    # Which row is kept for each pair, and where that pair first appeared. The two are
    # not the same position -- the best-scoring row can come later -- and it is the
    # first appearance that orders the output, since the order of the pairs is what a
    # candidate set of the search is indexed against.
    chosen = {}
    for position, pair in enumerate(zip(ppi_data[header_interactorA],
                                        ppi_data[header_interactorB])):
        previous = chosen.get(pair)
        if previous is None:
            chosen[pair] = (position, position)
            continue
        first, best = previous
        if scores is not None:
            if keep == 'highest':
                better = scores[position] > scores[best]
            else:
                better = scores[position] < scores[best]
            if better:
                chosen[pair] = (first, position)

    if verbose:
        print('Keeping {} of {} rows, one per interacting pair'
              .format(len(chosen), len(ppi_data)))

    kept = [best for _, best in sorted(chosen.values())]
    deduplicated_ppi = ppi_data.iloc[kept]
    return deduplicated_ppi.reset_index(drop=True)


def remove_ppi_bidirectionality(ppi_data, interaction_columns, verbose=True):
    '''
    Removes duplicate interactions. For example, when ProtA-ProtB and
    ProtB-ProtA interactions are present in the dataset, only one of
    them will be kept.

    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    interaction_columns : tuple
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    unidirectional_ppi : pandas.DataFrame
        List of protein-protein interactions without duplicated interactions
        in both directions (if ProtA-ProtB and ProtB-ProtA interactions are
        present, the lexicographically sorted orientation is the one kept, so
        'ProtA-ProtB' survives and 'ProtB-ProtA' is dropped). A one-way interaction
        whose reverse is absent is kept, and so is a self-interaction.
    '''
    if verbose:
        print('Removing bidirectionality of PPI network')
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]

    # Decided one unordered pair at a time. Collecting every protein that takes part in
    # some reciprocal interaction and then removing all combinations among them, as this
    # used to do, also deletes one-way interactions between two such proteins even when
    # their reverse was never in the table.
    #
    # Which of the two orientations survives is the lexicographic one, as it has always
    # been -- it decides which partner ends up as the ligand, so results carry over.
    pairs = set(zip(ppi_data[header_interactorA], ppi_data[header_interactorB]))
    drop = np.array([partner_a > partner_b and (partner_b, partner_a) in pairs
                     for partner_a, partner_b in zip(ppi_data[header_interactorA],
                                                     ppi_data[header_interactorB])],
                    dtype=bool)

    unidirectional_ppi = ppi_data.loc[~drop] if len(ppi_data) else ppi_data.copy()
    unidirectional_ppi = unidirectional_ppi.reset_index(drop=True)
    return unidirectional_ppi


def simplify_ppi(ppi_data, interaction_columns, score=None, verbose=True):
    '''
    Reduces a dataframe of protein-protein interactions into
    a simpler version with only three columns (A, B and score).

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    interaction_columns : tuple
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    score : str, default=None
        Column name where weights for the PPIs are specified. If
        None, a default score of one is automatically assigned
        to each PPI.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
        A simplified dataframe of protein-protein interactions with only
        three columns: 'A', 'B', 'score', wherein 'A' and 'B' are the
        interacting partners in the PPI and 'score' represents a weight
        of the interaction for computing cell-cell interactions/communication.
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
    '''
    Filters a list of protein-protein interactions to contain
    only interacting proteins in a list of specific protein or gene names.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    proteins : list
        A list of protein names to filter PPIs.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    upper_letter_comparison : boolean, default=True
        Whether making uppercase the protein names in the list of proteins and
        the names in the ppi_data to match their names and integrate their
        Useful when there are inconsistencies in the names that comes from a
        expression matrix and from ligand-receptor annotations.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    integrated_ppi : pandas.DataFrame
        A filtered list of PPIs by a given list of proteins or gene names.
    '''

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
    '''
    Makes a list of protein-protein interactions to be bidirectional.
    That is, repeating a PPI like ProtA-ProtB but in the other direction
    (ProtB-ProtA) if not present.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    interaction_columns : tuple
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    bi_ppi_data : pandas.DataFrame
        Bidirectional ppi_data. Contains duplicated PPIs in both directions.
        That is, it contains both ProtA-ProtB and ProtB-ProtA interactions.
    '''
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
    bi_ppi_data = pd.concat([ppi_A, ppi_B], join="inner")
    bi_ppi_data = bi_ppi_data.drop_duplicates()
    bi_ppi_data.reset_index(inplace=True, drop=True)
    return bi_ppi_data


def filter_ppi_network(ppi_data, contact_proteins, mediator_proteins=None, reference_list=None, bidirectional=True,
                       interaction_type='contacts', interaction_columns=('A', 'B'), verbose=True):
    '''
    Filters a list of protein-protein interactions to contain interacting
    proteins involved in different kinds of cell-cell interactions/communication.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    contact_proteins : list
        Protein names of proteins participating in cell contact
        interactions (e.g. surface proteins, receptors).

    mediator_proteins : list, default=None
        Protein names of proteins participating in mediated or
        secreted signaling (e.g. extracellular proteins, ligands).
        If None, only interactions involved in cell contacts
        will be returned.

    reference_list : list, default=None
        Reference list of protein names. Filtered PPIs from contact_proteins
        and mediator proteins will be keep only if those proteins are also
        present in this list when is not None.

    bidirectional : boolean, default=True
        Whether duplicating PPIs in both direction of interactions.
        That is, if the list considers ProtA-ProtB interaction,
        the interaction ProtB-ProtA will be also included.

    interaction_type : str, default='contacts'
        Type of intercellular interactions/communication where the proteins
        have to be involved in.
        Available types are:

        - 'contacts' : Contains proteins participating in cell contact
                interactions (e.g. surface proteins, receptors)
        - 'mediated' : Contains proteins participating in mediated or
                secreted signaling (e.g. ligand-receptor interactions)
        - 'combined' : Contains both 'contacts' and 'mediated' PPIs.
        - 'complete' : Contains all combinations of interactions between
                ligands, receptors, surface proteins, etc).

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    new_ppi_data : pandas.DataFrame
        A filtered list of PPIs by a given list of proteins or gene names
        depending on the type of intercellular communication.
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
    '''
    Gets protein/gene names for individual proteins (subunits when in complex)
    in a list of PPIs. If protein is a complex, for example ProtA&ProtB, it will
    return ProtA and ProtB separately.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    col_a_genes : list
        List of protein/gene names for proteins and subunits in the first column
        of interacting partners.

    complex_a : list
        List of list of subunits of each complex that were present in the first
        column of interacting partners and that were returned as subunits in the
        previous list.

    col_b_genes : list
        List of protein/gene names for proteins and subunits in the second column
        of interacting partners.

    complex_b : list
        List of list of subunits of each complex that were present in the second
        column of interacting partners and that were returned as subunits in the
        previous list.

    complexes : dict
        Dictionary where keys are the complex names in the list of PPIs, while
        values are list of subunits for the respective complex names.
    '''
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
    '''
    Filters a list of protein-protein interactions that for sure contains
    protein complexes to contain only interacting proteins or subunites
    in a list of specific protein or gene names.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    proteins : list
        A list of protein names to filter PPIs.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    upper_letter_comparison : boolean, default=True
        Whether making uppercase the protein names in the list of proteins and
        the names in the ppi_data to match their names and integrate their
        Useful when there are inconsistencies in the names that comes from a
        expression matrix and from ligand-receptor annotations.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    integrated_ppi : pandas.DataFrame
        A filtered list of PPIs, containing protein complexes in some cases,
        by a given list of proteins or gene names.
    '''
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
    '''
    Filters a list of protein-protein interactions to contain interacting
    proteins involved in different kinds of cell-cell interactions/communication.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    contact_proteins : list
        Protein names of proteins participating in cell contact
        interactions (e.g. surface proteins, receptors).

    mediator_proteins : list, default=None
        Protein names of proteins participating in mediated or
        secreted signaling (e.g. extracellular proteins, ligands).
        If None, only interactions involved in cell contacts
        will be returned.

    reference_list : list, default=None
        Reference list of protein names. Filtered PPIs from contact_proteins
        and mediator proteins will be keep only if those proteins are also
        present in this list when is not None.

    interaction_type : str, default='contacts'
        Type of intercellular interactions/communication where the proteins
        have to be involved in.
        Available types are:

        - 'contacts' : Contains proteins participating in cell contact
                interactions (e.g. surface proteins, receptors)
        - 'mediated' : Contains proteins participating in mediated or
                secreted signaling (e.g. ligand-receptor interactions)
        - 'combined' : Contains both 'contacts' and 'mediated' PPIs.
        - 'complete' : Contains all combinations of interactions between
                ligands, receptors, surface proteins, etc).

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    new_ppi_data : pandas.DataFrame
        A filtered list of PPIs by a given list of proteins or gene names
        depending on the type of intercellular communication.
    '''
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
        total_proteins = natsorted(set(contact_proteins + mediator_proteins))

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
    '''
    Filters a list of protein-protein interactions to
    contain only proteins in a given list in both
    columns of interacting partners.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    proteins : list
        A list of protein names to filter PPIs.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    new_ppi_data : pandas.DataFrame
        A filtered list of PPIs by a given list of proteins or gene names.
    '''
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]
    new_ppi_data = ppi_data.loc[ppi_data[header_interactorA].isin(proteins) & ppi_data[header_interactorB].isin(proteins)]
    new_ppi_data = new_ppi_data.drop_duplicates()
    return new_ppi_data


def get_one_group_to_other_ppi(ppi_data, proteins_a, proteins_b, interaction_columns=('A', 'B')):
    '''Filters a list of protein-protein interactions to
    contain specific proteins in the first column of
    interacting partners an other specific proteins in
    the second column.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    proteins_a : list
        A list of protein names to filter the first column of interacting
        proteins in a list of PPIs.

    proteins_b : list
        A list of protein names to filter the second column of interacting
        proteins in a list of PPIs.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    new_ppi_data : pandas.DataFrame
        A filtered list of PPIs by a given lists of proteins or gene names.
    '''
    header_interactorA = interaction_columns[0]
    header_interactorB = interaction_columns[1]
    direction1 = ppi_data.loc[ppi_data[header_interactorA].isin(proteins_a) & ppi_data[header_interactorB].isin(proteins_b)]
    direction2 = ppi_data.loc[ppi_data[header_interactorA].isin(proteins_b) & ppi_data[header_interactorB].isin(proteins_a)]
    direction2.columns = [header_interactorB, header_interactorA, 'score']
    direction2 = direction2[[header_interactorA, header_interactorB, 'score']]
    new_ppi_data = pd.concat([direction1, direction2], ignore_index=True).drop_duplicates()
    return new_ppi_data


def bidirectional_ppi_with_index(ppi_data, interaction_columns=('A', 'B'), verbose=False):
    '''
    Builds the bidirectional PPI table and records where each of its rows came from.

    The table is doubled because the undirected CCI scores consider ligands and
    receptors of **both** interacting cells, so an interaction has to be available in
    either direction. Self-interactions are the exception: a protein pairing with
    itself yields the same directed row twice, and keeping both would count it
    double, so duplicates are dropped.

    Unlike calling `bidirectional_ppi_for_cci` and working out the correspondence
    afterwards, the provenance here is exact **by construction**, so it does not have
    to be inferred from the values in any column.

    Rows are dropped on the same rule as `bidirectional_ppi_for_cci`, whole-row
    equality, so two rows listing the same partners with different metadata -- a
    different score, a different annotation -- are both kept, and each keeps its own
    place in `origin`.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs).

    interaction_columns : tuple, default=('A', 'B')
        Columns holding the two partners.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    Returns
    -------
    bi_ppi_data : pandas.DataFrame
        The bidirectional table, identical to what `bidirectional_ppi_for_cci`
        returns for the same input.

    origin : numpy.ndarray
        For each row of `bi_ppi_data`, the position in `ppi_data` it came from. Use
        it to expand a per-interaction vector -- a candidate selection, a weight, a
        permutation -- to the rows the interaction space expects.
    '''
    if verbose:
        print("Making bidirectional PPI for CCI.")
    prot_a, prot_b = interaction_columns

    swapped = ppi_data.copy()
    swapped[prot_a] = ppi_data[prot_b].values
    swapped[prot_b] = ppi_data[prot_a].values

    stacked = pd.concat([ppi_data, swapped], ignore_index=True)
    origin = np.concatenate([np.arange(len(ppi_data)), np.arange(len(ppi_data))])

    # A self-interaction appears identically in both halves; keeping one copy is what
    # stops it being counted twice. Whole-row equality rather than the interacting
    # columns alone, so a pair listed twice with different metadata survives as two
    # rows, exactly as `bidirectional_ppi_for_cci` leaves it.
    keep = ~stacked.duplicated(keep='first').values
    if verbose:
        print("Removing duplicates in bidirectional PPI network.")
    return stacked.loc[keep].reset_index(drop=True), origin[keep]


def bidirectional_index(ppi_data, interaction_columns=('A', 'B'), verbose=False):
    '''
    Position in `ppi_data` that each row of its bidirectional table comes from.

    Convenience wrapper over `bidirectional_ppi_with_index` for callers that already
    hold the bidirectional table and only need the mapping -- for instance to expand
    a candidate selection of ligand-receptor pairs into the per-row weights an
    interaction space built on that table expects.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs).

    interaction_columns : tuple, default=('A', 'B')
        Columns holding the two partners.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    Returns
    -------
    origin : numpy.ndarray
        One entry per row of the bidirectional table.
    '''
    return bidirectional_ppi_with_index(ppi_data,
                                        interaction_columns=interaction_columns,
                                        verbose=verbose)[1]
