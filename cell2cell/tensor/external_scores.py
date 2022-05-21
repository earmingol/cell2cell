# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from collections import defaultdict
from cell2cell.tensor.tensor import PreBuiltTensor


def dataframes_to_tensor(context_df_dict, sender_col, receiver_col, ligand_col, receptor_col, score_col, how='inner',
                         lr_sep='^', context_order=None, order_labels=None, device=None):
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

    lr_sep : str, default='^'
        Separation character to join ligands and receptors into a LR pair name.

    context_order : list, default=None
        List used to sort the contexts when building the tensor. Elements must
        be all elements in context_df_dict.keys().

    order_labels : list, default=None
        List containing the labels for each order or dimension of the tensor. For
        example: ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

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
        context_order = list(context_df_dict.keys())
    else:
        assert all([c in context_df_dict.keys() for c in context_order]), "The list 'context_order' must contain all context names contained in the keys of 'context_dict'"
        assert len(context_order) == len(context_df_dict.keys()), "Each context name must be contained only once in the list 'context_order'"

    cols = [sender_col, receiver_col, ligand_col, receptor_col, score_col]
    assert all([c in df.columns for c in cols for df in context_df_dict.values()]), "All input columns must be contained in all dataframes included in 'context_dict'"

    # Copy context dict to make modifications
    cont_dict = {k : v.copy()[cols] for k, v in context_df_dict.items()}

    # Find all existing LR pairs, sender and receiver cells across contexts
    lr_dict = defaultdict(set)
    sender_dict = defaultdict(set)
    receiver_dict = defaultdict(set)

    for k, df in cont_dict.items():
        df['LRs'] = df.apply(lambda row: row[ligand_col] + lr_sep + row[receptor_col], axis=1)

        lr_dict[k].update(df['LRs'].unique().tolist())
        sender_dict[k].update(df[sender_col].unique().tolist())
        receiver_dict[k].update(df[receiver_col].unique().tolist())

    if order_labels is None:
        order_labels = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

    # Subset LR pairs, sender and receiver cells given parameter 'how'
    for i, k in enumerate(context_order):
        if i == 0:
            inter_lrs = set(lr_dict[k])
            inter_senders = set(sender_dict[k])
            inter_receivers = set(receiver_dict[k])

            union_lrs = set(lr_dict[k])
            union_senders = set(sender_dict[k])
            union_receivers = set(receiver_dict[k])

        else:
            inter_lrs = inter_lrs.intersection(set(lr_dict[k]))
            inter_senders = inter_senders.intersection(set(sender_dict[k]))
            inter_receivers = inter_receivers.intersection(set(receiver_dict[k]))

            union_lrs = union_lrs.union(set(lr_dict[k]))
            union_senders = union_senders.union(set(sender_dict[k]))
            union_receivers = union_receivers.union(set(receiver_dict[k]))

    if how == 'inner':
        lr_pairs = list(inter_lrs)
        sender_cells = list(inter_senders)
        receiver_cells = list(inter_receivers)
    elif how == 'outer':
        lr_pairs = list(union_lrs)
        sender_cells = list(union_senders)
        receiver_cells = list(union_receivers)
    elif how == 'outer_lrs':
        lr_pairs = list(union_lrs)
        sender_cells = list(inter_senders)
        receiver_cells = list(inter_receivers)
    elif how == 'outer_cells':
        lr_pairs = list(inter_lrs)
        sender_cells = list(union_senders)
        receiver_cells = list(union_receivers)
    else:
        raise ValueError("Not a valid input for parameter 'how'")

    # Build temporal tensor to pass to PreBuiltTensor
    tmp_tensor = []
    for k in context_order:
        v = cont_dict[k]
        # 3D tensor for the context
        tmp_3d_tensor = []
        for lr in lr_pairs:
            df = v.loc[v['LRs'] == lr]
            if df.shape[0] == 0:  # TODO: Check behavior when df is empty
                df = pd.DataFrame(np.nan, index=sender_cells, columns=receiver_cells)
            else:
                df = df.pivot(index=sender_col, columns=receiver_col, values=score_col)
                df = df.reindex(sender_cells).reindex(receiver_cells, axis='columns')

            tmp_3d_tensor.append(df.values)
        tmp_tensor.append(tmp_3d_tensor)

    # Create InteractionTensor using PreBuiltTensor
    tensor = np.asarray(tmp_tensor)
    if how != 'inner':
        mask = (~np.isnan(tensor)).astype(int)
    else:
        mask = None
    interaction_tensor = PreBuiltTensor(tensor=np.nan_to_num(tensor),
                                        order_names=[context_order, lr_pairs, sender_cells, receiver_cells],
                                        order_labels=order_labels,
                                        mask=mask,
                                        device=device)
    return interaction_tensor