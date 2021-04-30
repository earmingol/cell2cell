# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorly as tl

from collections import OrderedDict

from cell2cell.core.communication_scores import compute_ccc_matrix
from cell2cell.preprocessing.ppi import get_genes_from_complexes
from cell2cell.preprocessing.rnaseq import add_complexes_to_expression
from cell2cell.preprocessing.ppi import filter_ppi_by_proteins
from cell2cell.tensor.factorization import _compute_tensor_factorization, _run_elbow_analysis, _multiple_runs_elbow_analysis, _compute_elbow
from cell2cell.plotting.tensor_factors_plot import plot_elbow, plot_multiple_run_elbow


class BaseTensor():
    def __init__(self):
        # Save variables for this class
        self.communication_score = None
        self.how = None
        self.tensor = None
        self.genes = None
        self.cells = None
        self.order_names = [None, None, None, None]
        self.tl_object = None
        self.factors = None
        self.rank = None
        self.mask = None

    def compute_tensor_factorization(self, rank, tf_type='non_negative_cp', init='svd', random_state=None, verbose=False,
                                     **kwargs):
        tf = _compute_tensor_factorization(tensor=self.tensor,
                                           rank=rank,
                                           tf_type=tf_type,
                                           init=init,
                                           random_state=random_state,
                                           mask=self.mask,
                                           verbose=verbose,
                                           **kwargs)

        self.tl_object = tf
        factor_names = ['Factor {}'.format(i) for i in range(1, rank+1)]
        order_names = ['Context', 'LRs', 'Sender', 'Receiver']
        self.factors = OrderedDict(zip(order_names,
                                       [pd.DataFrame(f, index=idx, columns=factor_names) for f, idx in zip(tf.factors, self.order_names)]))
        self.rank = rank


    def elbow_rank_selection(self, upper_rank=50, runs=20, tf_type='non_negative_cp', init='random', random_state=None,
                             automatic_elbow=True, mask=None, ci='std', figsize=(4, 2.25), fontsize=14, filename=None,
                             verbose=False, **kwargs):
        # Run analysis
        if verbose:
            print('Running Elbow Analysis')

        if runs == 1:
            loss = _run_elbow_analysis(tensor=self.tensor,
                                       upper_rank=upper_rank,
                                       tf_type=tf_type,
                                       init=init,
                                       random_state=random_state,
                                       mask=mask,
                                       verbose=verbose,
                                       **kwargs
                                       )
            if automatic_elbow:
                rank = _compute_elbow(loss)
            else:
                rank = None
            fig = plot_elbow(loss=loss,
                             elbow=rank,
                             figsize=figsize,
                             fontsize=fontsize,
                             filename=filename)
        elif runs > 1:

            all_loss = _multiple_runs_elbow_analysis(tensor=self.tensor,
                                                     upper_rank=upper_rank,
                                                     runs=runs,
                                                     tf_type=tf_type,
                                                     init=init,
                                                     random_state=random_state,
                                                     mask=mask,
                                                     verbose=verbose,
                                                     **kwargs
                                                     )

            # Same outputs as runs = 1
            loss = np.nanmean(all_loss, axis=0).tolist()
            loss = [(i + 1, l) for i, l in enumerate(loss)]

            if automatic_elbow:
                rank = _compute_elbow(loss)
            else:
                rank = None

            fig = plot_multiple_run_elbow(all_loss=all_loss,
                                          ci=ci,
                                          elbow=rank,
                                          figsize=figsize,
                                          fontsize=fontsize,
                                          filename=filename)

        else:
            assert runs > 0, "Input runs must be an integer greater than 0"

        if automatic_elbow:
            self.rank = rank
            print('The rank at the elbow is: {}'.format(self.rank))
        return fig, loss


    def get_top_factor_elements(self, order_name, factor_name, top_number=10):
        top_elements = self.factors[order_name][factor_name].sort_values(ascending=False).head(top_number)
        return top_elements

    def export_factor_loadings(self, filename):
        writer = pd.ExcelWriter(filename)
        for k, v in self.factors.items():
            v.to_excel(writer, sheet_name=k)
        writer.save()
        print('Loadings of the tensor factorization were successfully saved into {}'.format(filename))


class InteractionTensor(BaseTensor):

    def __init__(self, rnaseq_matrices, ppi_data, context_names=None, how='inner', communication_score='expression_product',
                 complex_sep=None, upper_letter_comparison=True, interaction_columns=('A', 'B'), verbose=True):
        # Init BaseTensor
        BaseTensor.__init__(self)

        # Generate expression values for protein complexes in PPI data
        if complex_sep is not None:
            if verbose:
                print('Getting expression values for protein complexes')
            col_a_genes, complex_a, col_b_genes, complex_b, complexes = get_genes_from_complexes(ppi_data=ppi_data,
                                                                                                 complex_sep=complex_sep,
                                                                                                 interaction_columns=interaction_columns
                                                                                                 )
            mod_rnaseq_matrices = [add_complexes_to_expression(rnaseq, complexes) for rnaseq in rnaseq_matrices]
        else:
            mod_rnaseq_matrices = [df.copy() for df in rnaseq_matrices]

        # Uppercase for Gene names
        if upper_letter_comparison:
            for df in mod_rnaseq_matrices:
                df.index = [idx.upper() for idx in df.index]


        # Get context CCC tensor
        tensor, genes, cells, ppi_names, mask = build_context_ccc_tensor(rnaseq_matrices=mod_rnaseq_matrices,
                                                                         ppi_data=ppi_data,
                                                                         how=how,
                                                                         communication_score=communication_score,
                                                                         complex_sep=complex_sep,
                                                                         upper_letter_comparison=upper_letter_comparison,
                                                                         interaction_columns=interaction_columns,
                                                                         verbose=verbose)

        # Generate names for the elements in each dimension (order) in the tensor
        if context_names is None:
            context_names = ['C-' + str(i) for i in range(1, len(mod_rnaseq_matrices)+1)]
            # for PPIS use ppis, and sender & receiver cells, use cells

        # Save variables for this class
        self.communication_score = communication_score
        self.how = how
        self.tensor = tl.tensor(tensor)
        self.genes = genes
        self.cells = cells
        self.order_names = [context_names, ppi_names, self.cells, self.cells]
        self.mask = mask


class PreBuiltTensor(BaseTensor):
    def __init__(self, tensor, order_names):
        # Init BaseTensor
        BaseTensor.__init__(self)

        self.tensor = tl.tensor(tensor)
        self.order_names = order_names


def build_context_ccc_tensor(rnaseq_matrices, ppi_data, how='inner', communication_score='expression_product',
                             complex_sep=None, upper_letter_comparison=True, interaction_columns=('A', 'B'), verbose=True):

    df_idxs = [list(rnaseq.index) for rnaseq in rnaseq_matrices]
    df_cols = [list(rnaseq.columns) for rnaseq in rnaseq_matrices]
    if how == 'inner':
        genes = set.intersection(*map(set, df_idxs))
        cells = set.intersection(*map(set, df_cols))
    elif how == 'outer':
        genes = set.union(*map(set, df_idxs))
        cells = set.union(*map(set, df_cols))
    else:
        raise ValueError('Provide a valid way to build the tensor; "how" must be "inner" or "outer" ')

    # Preserve order or sort new set (either inner or outer)
    if set(df_idxs[0]) == genes:
        genes = df_idxs[0]
    else:
        genes = sorted(list(genes))

    if set(df_cols[0]) == cells:
        cells = df_cols[0]
    else:
        cells = sorted(list(cells))

    # Filter PPI data for
    ppi_data_ = filter_ppi_by_proteins(ppi_data=ppi_data,
                                       proteins=genes,
                                       complex_sep=complex_sep,
                                       upper_letter_comparison=upper_letter_comparison,
                                       interaction_columns=interaction_columns)

    if verbose:
        print('Building tensor for the provided context')

    tensors = [generate_ccc_tensor(rnaseq_data=rnaseq.reindex(genes, fill_value=0.).reindex(cells, axis='columns', fill_value=0.),
                                   ppi_data=ppi_data_,
                                   communication_score=communication_score,
                                   interaction_columns=interaction_columns) for rnaseq in rnaseq_matrices]

    # Generate mask:
    if how == 'outer':
        mask_tensor = []
        for rnaseq in rnaseq_matrices:
            rna_cells = list(rnaseq.columns)
            ppi_mask = pd.DataFrame(np.ones((len(rna_cells), len(rna_cells))), columns=rna_cells, index=rna_cells)
            ppi_mask = ppi_mask.reindex(cells, fill_value=0.).reindex(cells, axis='columns', fill_value=0.) # Here we mask those cells that are not in the rnaseq data
            rna_tensor = [ppi_mask.values for i in range(ppi_data_.shape[0])] # Replicate the mask across all PPIs in ppi_data_
            mask_tensor.append(rna_tensor)
        mask_tensor = np.asarray(mask_tensor)
    else:
        mask_tensor = None

    ppi_names = [row[interaction_columns[0]] + '^' + row[interaction_columns[1]] for idx, row in ppi_data_.iterrows()]
    return tensors, genes, cells, ppi_names, mask_tensor


def generate_ccc_tensor(rnaseq_data, ppi_data, communication_score='expression_product', interaction_columns=('A', 'B')):
    ppi_a = interaction_columns[0]
    ppi_b = interaction_columns[1]

    ccc_tensor = []
    for idx, ppi in ppi_data.iterrows():
        v = rnaseq_data.loc[ppi[ppi_a], :].values
        w = rnaseq_data.loc[ppi[ppi_b], :].values
        ccc_tensor.append(compute_ccc_matrix(prot_a_exp=v,
                                             prot_b_exp=w,
                                             communication_score=communication_score).tolist())
    return ccc_tensor


def generate_tensor_metadata(interaction_tensor, metadata_dicts, fill_with_order_elements=True):
    '''
    Uses a list of of dicts (or None when a dict is missing) to generate a list of metadata for each order in the tensor.

    Parameters
    ----------
    interaction_tensor : cell2cell.tensor.InteractionTensor class

    metadata_dicts : list
        A list of dictionaries. Each dictionary represents an order of the tensor. In an interaction tensor these orders
        should be contexts, LR pairs, sender cells and receiver cells. The keys are the elements in each order
        (they are contained in interaction_tensor.order_names) and the values are the categories that each elements will
        be assigned as metadata.

    fill_with_order_elements : boolean, True by default
        Whether using each element of a dimension as its own metadata when a None is passed instead of a dictionary for
        the respective order/dimension. If True, each element in that order will be use itself, that dimension will not
        contain metadata.

    Returns
    -------
    metadata : list
        A list of pandas.DataFrames that will be used as an input of the cell2cell.plot.tensor_factors_plot.
    '''
    assert (len(interaction_tensor.tensor.shape) == len(metadata_dicts)), "metadata_dicts should be of the same size as the number of orders/dimensions in the tensor"

    if fill_with_order_elements:
        metadata = [pd.DataFrame(index=names) for names in interaction_tensor.order_names]
        metadata[1] = pd.DataFrame(index=["Ligand-Receptor Pairs"]*len(metadata[1])) # To avoid having thousands of different categories (each LR pair would be one)
    else:
        metadata = [pd.DataFrame(index=names) if (meta is not None) else None for names, meta in zip(interaction_tensor.order_names, metadata_dicts)]

    for i, meta in enumerate(metadata):
        if meta is not None:
            if metadata_dicts[i] is not None:
                meta['Category'] = [metadata_dicts[i][idx] for idx in meta.index]
            else:
                meta['Category'] = meta.index
            meta.index.name = 'Element'
            meta.reset_index(inplace=True)
    return metadata


def interactions_to_tensor(interactions, experiment='single_cell', context_names=None, how='inner',
                           communication_score='expression_product', upper_letter_comparison=True, verbose=True):
    '''Experiment either "bulk" or "single_cell'''
    ppis = []
    rnaseq_matrices = []
    complex_sep = interactions[0].complex_sep
    interaction_columns = interactions[0].interaction_columns
    for Int_ in interactions:
        if Int_.analysis_setup['cci_type'] == 'undirected':
            ppis.append(Int_.ref_ppi)
        else:
            ppis.append(Int_.ppi_data)

        if experiment == 'single_cell':
            rnaseq_matrices.append(Int_.aggregated_expression)
        elif experiment == 'bulk':
            rnaseq_matrices.append(Int_.rnaseq_data)
        else:
            raise ValueError("experiment must be 'single_cell' or 'bulk'")

    ppi_data = pd.concat(ppis)
    ppi_data = ppi_data.drop_duplicates().reset_index(drop=True)

    tensor = InteractionTensor(rnaseq_matrices=rnaseq_matrices,
                               ppi_data=ppi_data,
                               context_names=context_names,
                               how=how,
                               complex_sep=complex_sep,
                               interaction_columns=interaction_columns,
                               communication_score=communication_score,
                               upper_letter_comparison=upper_letter_comparison,
                               verbose=verbose
                               )
    return tensor