# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.preprocessing import ppi, gene_ontology, rnaseq


## RNAseq datasets
def get_modified_rnaseq(rnaseq_data, cutoffs=None, communication_score='expression_thresholding'):
    '''
    Preprocess gene expression into values used by a communication
    scoring function (either continuous or binary).

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    cutoffs : pandas.DataFrame
        A dataframe containing the value corresponding to cutoff or threshold
        assigned to each gene. Rows are genes and columns could be either
        'value' for a single threshold for all cell-types/tissues/samples
        or the names of cell-types/tissues/samples for thresholding in a
        specific way.
        They could be obtained through the function
        cell2cell.preprocessing.cutoffs.get_cutoffs()

    communication_score : str, default='expression_thresholding'
        Type of communication score used to detect active ligand-receptor
        pairs between each pair of cell. See
        cell2cell.core.communication_scores for more details.
        It can be:

        - 'expression_thresholding'
        - 'expression_product'
        - 'expression_mean'
        - 'expression_gmean'

    Returns
    -------
    modified_rnaseq : pandas.DataFrame
        Preprocessed gene expression given a communication scoring
        function to use. Rows are genes and columns are
        cell-types/tissues/samples.
    '''
    if communication_score == 'expression_thresholding':
        modified_rnaseq = get_thresholded_rnaseq(rnaseq_data, cutoffs)
    elif communication_score in ['expression_product', 'expression_mean', 'expression_gmean']:
        modified_rnaseq = rnaseq_data.copy()
    else:
        # As other score types are implemented, other elif condition will be included here.
        raise NotImplementedError("Score type {} to compute pairwise cell-interactions is not implemented".format(communication_score))
    return modified_rnaseq


def get_thresholded_rnaseq(rnaseq_data, cutoffs):
    '''Binzarizes a RNA-seq dataset given cutoff or threshold
    values.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    cutoffs : pandas.DataFrame
        A dataframe containing the value corresponding to cutoff or threshold
        assigned to each gene. Rows are genes and columns could be either
        'value' for a single threshold for all cell-types/tissues/samples
        or the names of cell-types/tissues/samples for thresholding in a
        specific way.
        They could be obtained through the function
        cell2cell.preprocessing.cutoffs.get_cutoffs()

    Returns
    -------
    binary_rnaseq_data : pandas.DataFrame
        Preprocessed gene expression into binary values given cutoffs
        or thresholds either general or specific for all cell-types/
        tissues/samples.
    '''
    binary_rnaseq_data = rnaseq_data.copy()
    columns = list(cutoffs.columns)
    if (len(columns) == 1) and ('value' in columns):
        binary_rnaseq_data = binary_rnaseq_data.gt(list(cutoffs.value.values), axis=0)
    elif sorted(columns) == sorted(list(rnaseq_data.columns)):  # Check there is a column for each cell type
        for col in columns:
            binary_rnaseq_data[col] = binary_rnaseq_data[col].gt(list(cutoffs[col].values), axis=0) # ge
    else:
        raise KeyError("The cutoff data provided does not have a 'value' column or does not match the columns in rnaseq_data.")
    binary_rnaseq_data = binary_rnaseq_data.astype(float)
    return binary_rnaseq_data


## PPI datasets
def get_weighted_ppi(ppi_data, modified_rnaseq_data, column='value', interaction_columns=('A', 'B')):
    '''
    Assigns preprocessed gene expression values to
    proteins in a list of protein-protein interactions.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    modified_rnaseq_data : pandas.DataFrame
        Preprocessed gene expression given a communication scoring
        function to use. Rows are genes and columns are
        cell-types/tissues/samples.

    column : str, default='value'
        Column name to consider the gene expression values.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    weighted_ppi : pandas.DataFrame
        List of protein-protein interactions that contains gene expression
        values instead of the names of interacting proteins. Gene expression
        values are preprocessed given a communication scoring function to
        use.
    '''
    prot_a = interaction_columns[0]
    prot_b = interaction_columns[1]
    weighted_ppi = ppi_data.copy()
    weighted_ppi[prot_a] = weighted_ppi[prot_a].apply(func=lambda row: modified_rnaseq_data.at[row, column]) # Replaced .loc by .at
    weighted_ppi[prot_b] = weighted_ppi[prot_b].apply(func=lambda row: modified_rnaseq_data.at[row, column])
    weighted_ppi = weighted_ppi[[prot_a, prot_b, 'score']].reset_index(drop=True).fillna(0.0)
    return weighted_ppi


def get_ppi_dict_from_proteins(ppi_data, contact_proteins, mediator_proteins=None, interaction_columns=('A', 'B'),
                               bidirectional=True, verbose=True):
    '''
    Filters a complete list of protein-protein interactions into
    sublists containing proteins involved in different kinds of
    intercellular interactions, by provided lists of proteins.

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

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    bidirectional : boolean, default=True
        Whether duplicating PPIs in both direction of interactions.
        That is, if the list considers ProtA-ProtB interaction,
        the interaction ProtB-ProtA will be also included.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    ppi_dict : dict
        Dictionary containing lists of PPIs involving proteins that
        participate in diffferent kinds of intercellular interactions.
        Options are under the keys:

        - 'contacts' : Contains proteins participating in cell contact
                interactions (e.g. surface proteins, receptors)
        - 'mediated' : Contains proteins participating in mediated or
                secreted signaling (e.g. ligand-receptor interactions)
        - 'combined' : Contains both 'contacts' and 'mediated' PPIs.
        - 'complete' : Contains all combinations of interactions between
                ligands, receptors, surface proteins, etc).
        If mediator_proteins input is None, this dictionary will contain
        PPIs only for 'contacts'.
    '''


    ppi_dict = dict()
    ppi_dict['contacts'] = ppi.filter_ppi_network(ppi_data=ppi_data,
                                                  contact_proteins=contact_proteins,
                                                  mediator_proteins=mediator_proteins,
                                                  interaction_type='contacts',
                                                  interaction_columns=interaction_columns,
                                                  bidirectional=bidirectional,
                                                  verbose=verbose)
    if mediator_proteins is not None:
        for interaction_type in ['mediated', 'combined', 'complete']:
            ppi_dict[interaction_type] = ppi.filter_ppi_network(ppi_data=ppi_data,
                                                          contact_proteins=contact_proteins,
                                                          mediator_proteins=mediator_proteins,
                                                          interaction_type=interaction_type,
                                                          interaction_columns=interaction_columns,
                                                          bidirectional=bidirectional,
                                                          verbose=verbose)

    return ppi_dict


def get_ppi_dict_from_go_terms(ppi_data, go_annotations, go_terms, contact_go_terms, mediator_go_terms=None, use_children=True,
                               go_header='GO', gene_header='Gene', interaction_columns=('A', 'B'), verbose=True):
    '''
    Filters a complete list of protein-protein interactions into
    sublists containing proteins involved in different kinds of
    intercellular interactions, by provided lists of GO terms.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    go_annotations : pandas.DataFrame
        Dataframe containing information about GO term annotations of each
        gene for a given organism according to the ga file. Can be loading
        with the function cell2cell.io.read_data.load_go_annotations().

    go_terms : networkx.Graph
        NetworkX Graph containing GO terms datasets from .obo file.
        It could be loaded using
        cell2cell.io.read_data.load_go_terms(filename).

    contact_go_terms : list
        GO terms for selecting proteins participating in cell contact
        interactions (e.g. surface proteins, receptors).

    mediator_go_terms : list, default=None
        GO terms for selecting proteins participating in mediated or
        secreted signaling (e.g. extracellular proteins, ligands).
        If None, only interactions involved in cell contacts
        will be returned.

    use_children : boolean, default=True
        Whether considering children GO terms (below in hierarchy) to the
        ones passed as inputs (contact_go_terms and mediator_go_terms).

    go_header : str, default='GO'
        Column name wherein GO terms are located in the dataframe.

    gene_header : str, default='Gene'
        Column name wherein genes are located in the dataframe.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    verbose : boolean, default=True
        Whether printing or not steps of the analysis.

    Returns
    -------
    ppi_dict : dict
        Dictionary containing lists of PPIs involving proteins that
        participate in diffferent kinds of intercellular interactions.
        Options are under the keys:

        - 'contacts' : Contains proteins participating in cell contact
                interactions (e.g. surface proteins, receptors)
        - 'mediated' : Contains proteins participating in mediated or
                secreted signaling (e.g. ligand-receptor interactions)
        - 'combined' : Contains both 'contacts' and 'mediated' PPIs.
        - 'complete' : Contains all combinations of interactions between
                ligands, receptors, surface proteins, etc).
        If mediator_go_terms input is None, this dictionary will contain
        PPIs only for 'contacts'.
    '''
    if use_children == True:
        contact_proteins = gene_ontology.get_genes_from_go_hierarchy(go_annotations=go_annotations,
                                                                     go_terms=go_terms,
                                                                     go_filter=contact_go_terms,
                                                                     go_header=go_header,
                                                                     gene_header=gene_header,
                                                                     verbose=verbose)

        mediator_proteins = gene_ontology.get_genes_from_go_hierarchy(go_annotations=go_annotations,
                                                                      go_terms=go_terms,
                                                                      go_filter=mediator_go_terms,
                                                                      go_header=go_header,
                                                                      gene_header=gene_header,
                                                                      verbose=verbose)
    else:
        contact_proteins = gene_ontology.get_genes_from_go_terms(go_annotations=go_annotations,
                                                                 go_filter=contact_go_terms,
                                                                 go_header=go_header,
                                                                 gene_header=gene_header,
                                                                 verbose=verbose)

        mediator_proteins = gene_ontology.get_genes_from_go_terms(go_annotations=go_annotations,
                                                                  go_filter=mediator_go_terms,
                                                                  go_header=go_header,
                                                                  gene_header=gene_header,
                                                                  verbose=verbose)

    # Avoid same genes in list
    #contact_proteins = list(set(contact_proteins) - set(mediator_proteins))

    ppi_dict = get_ppi_dict_from_proteins(ppi_data=ppi_data,
                                          contact_proteins=contact_proteins,
                                          mediator_proteins=mediator_proteins,
                                          interaction_columns=interaction_columns,
                                          verbose=verbose)

    return ppi_dict