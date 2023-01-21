# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from collections import defaultdict
from tqdm.auto import tqdm
from cell2cell.preprocessing.find_elements import get_element_abundances, get_elements_over_fraction
from cell2cell.tensor.tensor import PreBuiltTensor


def dataframes_to_tensor(context_df_dict, sender_col, receiver_col, ligand_col, receptor_col, score_col, how='inner',
                         outer_fraction=0.0, lr_fill=np.nan, cell_fill=np.nan, lr_sep='^', dup_aggregation='max',
                         context_order=None, order_labels=None, sort_elements=True, device=None):
    '''Generates an InteractionTensor from a dictionary
    containing dataframes for all contexts.

    Parameters
    ----------
    context_df_dict : dict
        Dictionary containing a dataframe for each context. The dataframe
        must contain columns containing sender cells, receiver cells,
        ligands, receptors, and communication scores, separately.
        Keys are context names and values are dataframes.

    sender_col : str
        Name of the column containing the sender cells in all context
        dataframes.

    receiver_col : str
        Name of the column containing the receiver cells in all context
        dataframes.

    ligand_col : str
        Name of the column containing the ligands in all context
        dataframes.

    receptor_col : str
        Name of the column containing the receptors in all context
        dataframes.

    score_col : str
        Name of the column containing the communication scores in all context
        dataframes.

    how : str, default='inner'
        Approach to consider cell types and genes present across multiple contexts.

        - 'inner' : Considers only cell types and LR pairs that are present in all
                    contexts (intersection).
        - 'outer' : Considers all cell types and LR pairs that are present
                    across contexts (union).
        - 'outer_lrs' : Considers only cell types that are present in all
                        contexts (intersection), while all LR pairs that are
                        present across contexts (union).
        - 'outer_cells' : Considers only LR pairs that are present in all
                          contexts (intersection), while all cell types that are
                          present across contexts (union).

    outer_fraction : float, default=0.0
        Threshold to filter the elements when `how` includes any outer option.
        Elements with a fraction abundance across contexts (in `context_df_dict`)
        at least this threshold will be included. When this value is 0, considers
        all elements across the samples. When this value is 1, it acts as using
        `how='inner'`.

    lr_fill : float, default=numpy.nan
        Value to fill communication scores when a ligand-receptor pair is not
        present across all contexts.

    cell_fill : float, default=numpy.nan
        Value to fill communication scores when a cell is not
        present across all ligand-receptor pairs or all contexts.

    lr_sep : str, default='^'
        Separation character to join ligands and receptors into a LR pair name.

    dup_aggregation : str, default='max'
        Approach to aggregate communication score if there are multiple instances
        of an LR pair for a specific sender-receiver pair in one of the dataframes.

        - 'max' : Maximum of the multiple instances
        - 'min' : Minimum of the multiple instances
        - 'mean' : Average of the multiple instances
        - 'median' : Median of the multiple instances

    context_order : list, default=None
        List used to sort the contexts when building the tensor. Elements must
        be all elements in context_df_dict.keys().

    order_labels : list, default=None
        List containing the labels for each order or dimension of the tensor. For
        example: ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

    sort_elements : boolean, default=True
        Whether alphabetically sorting elements in the InteractionTensor.
        The Context Dimension is not sorted if a 'context_order' list is provided.

    device : str, default=None
            Device to use when backend is pytorch. Options are:
             {'cpu', 'cuda:0', None}

    Returns
    -------
    interaction_tensor : cell2cell.tensor.PreBuiltTensor
        A communication tensor generated for the Tensor-cell2cell pipeline.
    '''
    # Make sure that all contexts contain needed info
    if context_order is None:
        sort_context = sort_elements
        context_order = list(context_df_dict.keys())
    else:
        assert all([c in context_df_dict.keys() for c in context_order]), "The list 'context_order' must contain all context names contained in the keys of 'context_dict'"
        assert len(context_order) == len(context_df_dict.keys()), "Each context name must be contained only once in the list 'context_order'"
        sort_context = False
    cols = [sender_col, receiver_col, ligand_col, receptor_col, score_col]
    assert all([c in df.columns for c in cols for df in context_df_dict.values()]), "All input columns must be contained in all dataframes included in 'context_dict'"

    # Copy context dict to make modifications
    cont_dict = {k : v.copy()[cols] for k, v in context_df_dict.items()}

    # Labels for each dimension
    if order_labels is None:
        order_labels = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

    # Find all existing LR pairs, sender and receiver cells across contexts
    lr_dict = defaultdict(set)
    sender_dict = defaultdict(set)
    receiver_dict = defaultdict(set)

    for k, df in cont_dict.items():
        df['LRs'] = df.apply(lambda row: row[ligand_col] + lr_sep + row[receptor_col], axis=1)
        # This is to consider only LR pairs in each context that are present in all cell pairs. Disabled for now.
        # if how in ['inner', 'outer_cells']:
        #     ccc_df = df.pivot(index='LRs', columns=[sender_col, receiver_col], values=score_col)
        #     ccc_df = ccc_df.dropna(how='any')
        #     lr_dict[k].update(list(ccc_df.index))
        # else:
        lr_dict[k].update(df['LRs'].unique().tolist())
        sender_dict[k].update(df[sender_col].unique().tolist())
        receiver_dict[k].update(df[receiver_col].unique().tolist())

    # Subset LR pairs, sender and receiver cells given parameter 'how'
    df_lrs = [list(lr_dict[k]) for k in context_order]
    df_senders = [list(sender_dict[k]) for k in context_order]
    df_receivers  = [list(receiver_dict[k]) for k in context_order]

    if how == 'inner':
        lr_pairs = list(set.intersection(*map(set, df_lrs)))
        sender_cells = list(set.intersection(*map(set, df_senders)))
        receiver_cells = list(set.intersection(*map(set, df_receivers)))
    elif how == 'outer':
        lr_pairs = get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_lrs),
                                              fraction=outer_fraction)
        sender_cells = get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_senders),
                                              fraction=outer_fraction)
        receiver_cells = get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_receivers),
                                              fraction=outer_fraction)
    elif how == 'outer_lrs':
        lr_pairs = get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_lrs),
                                              fraction=outer_fraction)
        sender_cells = list(set.intersection(*map(set, df_senders)))
        receiver_cells = list(set.intersection(*map(set, df_receivers)))
    elif how == 'outer_cells':
        lr_pairs = list(set.intersection(*map(set, df_lrs)))
        sender_cells = get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_senders),
                                                  fraction=outer_fraction)
        receiver_cells = get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_receivers),
                                                    fraction=outer_fraction)
    else:
        raise ValueError("Not a valid input for parameter 'how'")

    if sort_elements:
        if sort_context:
            context_order = sorted(context_order)
        lr_pairs = sorted(lr_pairs)
        sender_cells = sorted(sender_cells)
        receiver_cells = sorted(receiver_cells)

    # Build temporal tensor to pass to PreBuiltTensor
    tmp_tensor = []
    for k in tqdm(context_order):
        v = cont_dict[k]
        # 3D tensor for the context
        tmp_3d_tensor = []
        for lr in lr_pairs:
            df = v.loc[v['LRs'] == lr]
            if df.shape[0] == 0:  # TODO: Check behavior when df is empty
                df = pd.DataFrame(lr_fill, index=sender_cells, columns=receiver_cells)
            else:
                if df[cols[:-1]].duplicated().any():
                    assert dup_aggregation in ['max', 'min', 'mean', 'median'], "Please use a valid option for `dup_aggregation`."
                    df = getattr(df.groupby(cols[:-1]), dup_aggregation)().reset_index()
                df = df.pivot(index=sender_col, columns=receiver_col, values=score_col)
                df = df.reindex(sender_cells, fill_value=cell_fill).reindex(receiver_cells, fill_value=cell_fill, axis='columns')

            tmp_3d_tensor.append(df.values)
        tmp_tensor.append(tmp_3d_tensor)

    # Create InteractionTensor using PreBuiltTensor
    tensor = np.asarray(tmp_tensor)
    if how != 'inner':
        mask = (~np.isnan(tensor)).astype(int)
        loc_nans = np.ones(tensor.shape, dtype=int) - mask
    else:
        mask = None
        loc_nans = np.zeros(tensor.shape, dtype=int)

    interaction_tensor = PreBuiltTensor(tensor=tensor,
                                        order_names=[context_order, lr_pairs, sender_cells, receiver_cells],
                                        order_labels=order_labels,
                                        mask=mask,
                                        loc_nans=loc_nans,
                                        device=device)
    return interaction_tensor