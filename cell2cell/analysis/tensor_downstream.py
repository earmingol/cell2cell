# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from cell2cell.stats import gini_coefficient


def get_joint_loadings(result, dim1, dim2, factor):
    """
    Creates the joint loading distribution between two tensor dimensions for a
    given factor output from decomposition.

    Parameters
    ----------
    result : any Tensor class in cell2cell.tensor.tensor or a dict
        Either a Tensor type or a dictionary which resulted from the tensor
        decomposition. If it is a dict, it should be the one in, for example,
        InteractionTensor.factors

    dim1 : str
        One of the tensor dimensions (options are in the keys of the dict,
        or interaction.factors.keys())

    dim2 : str
        A second tensor dimension (options are in the keys of the dict,
        or interaction.factors.keys())

    factor: str
        One of the factors output from the decomposition (e.g. 'Factor 1').

    Returns
    -------
    joint_dist : pandas.DataFrame
        Joint distribution of factor loadings for the specified dimensions.
        Rows correspond to elements in dim1 and columns to elements in dim2.
    """
    if hasattr(result, 'factors'):
        result = result.factors
        if result is None:
            raise ValueError('A tensor factorization must be run on the tensor before calling this function.')
    elif isinstance(result, dict):
        pass
    else:
        raise ValueError('result is not of a valid type. It must be an InteractionTensor or a dict.')

    assert dim1 in result.keys(), 'The specified dimension ' + dim1 + ' is not present in the tensor'
    assert dim2 in result.keys(), 'The specified dimension ' + dim2 + ' is not present in the tensor'

    vec1 = result[dim1][factor]
    vec2 = result[dim2][factor]

    # Calculate the outer product
    joint_dist = pd.DataFrame(data=np.outer(vec1, vec2),
                              index=vec1.index,
                              columns=vec2.index)

    return joint_dist


def get_factor_specific_ccc_networks(result, sender_label='Sender Cells', receiver_label='Receiver Cells'):
    '''
    Generates adjacency matrices for each of the factors
    obtained from a tensor decomposition. These matrices represent a
    cell-cell communication directed network.

    Parameters
    ----------
    result : any Tensor class in cell2cell.tensor.tensor or a dict
        Either a Tensor type or a dictionary which resulted from the tensor
        decomposition. If it is a dict, it should be the one in, for example,
        InteractionTensor.factors

    sender_label : str
        Label for the dimension of sender cells. Usually found in
        InteractionTensor.order_labels

    receiver_label : str
        Label for the dimension of receiver cells. Usually found in
        InteractionTensor.order_labels

    Returns
    -------
    networks : dict
        A dictionary containing a pandas.DataFrame for each of the factors
        (factor names are the keys of the dict). These dataframes are the
        adjacency matrices of the CCC networks.
    '''
    if hasattr(result, 'factors'):
        result = result.factors
        if result is None:
            raise ValueError('A tensor factorization must be run on the tensor before calling this function.')
    elif isinstance(result, dict):
        pass
    else:
        raise ValueError('result is not of a valid type. It must be an InteractionTensor or a dict.')

    factors = sorted(list(set(result[sender_label].columns) & set(result[receiver_label].columns)))

    networks = dict()
    for f in factors:
        networks[f] = get_joint_loadings(result=result,
                                         dim1=sender_label,
                                         dim2=receiver_label,
                                         factor=f
                                         )
    return networks


def flatten_factor_ccc_networks(networks, orderby='senders'):
    '''
    Flattens all adjacency matrices in the factor-specific
    cell-cell communication networks. It generates a matrix
    where rows are factors and columns are cell-cell pairs.

    Parameters
    ----------
    networks : dict
        A dictionary containing a pandas.DataFrame for each of the factors
        (factor names are the keys of the dict). These dataframes are the
        adjacency matrices of the CCC networks.

    orderby : str
        Order of the flatten cell-cell pairs. Options are 'senders' and
        'receivers'. 'senders' means to flatten the matrices in a way that
        all cell-cell pairs with a same sender cell are put next to each others.
        'receivers' means the same, but by considering the receiver cell instead.

    Returns
    -------
    flatten_networks : pandas.DataFrame
        A dataframe wherein rows contains a factor-specific network. Columns are
        the directed cell-cell pairs.
    '''
    senders = sorted(set.intersection(*[set(v.index) for v in networks.values()]))
    receivers = sorted(set.intersection(*[set(v.columns) for v in networks.values()]))

    if orderby == 'senders':
        cell_pairs = [s + ' --> ' + r for s in senders for r in receivers]
        flatten_order = 'C'
    elif orderby == 'receivers':
        cell_pairs = [s + ' --> ' + r for r in receivers for s in senders]
        flatten_order = 'F'
    else:
        raise ValueError("`orderby` must be either 'senders' or 'receivers'.")

    data = np.asarray([v.values.flatten(flatten_order) for v in networks.values()]).T
    flatten_networks = pd.DataFrame(data=data,
                                    index=cell_pairs,
                                    columns=list(networks.keys())
                                    )
    return flatten_networks


def compute_gini_coefficients(result, sender_label='Sender Cells', receiver_label='Receiver Cells'):
    '''
    Computes Gini coefficient on the distribution of edge weights
    in each factor-specific cell-cell communication network. Factors
    obtained from the tensor decomposition with Tensor-cell2cell.

    Parameters
    ----------
    result : any Tensor class in cell2cell.tensor.tensor or a dict
        Either a Tensor type or a dictionary which resulted from the tensor
        decomposition. If it is a dict, it should be the one in, for example,
        InteractionTensor.factors

    sender_label : str
        Label for the dimension of sender cells. Usually found in
        InteractionTensor.order_labels

    receiver_label : str
        Label for the dimension of receiver cells. Usually found in
        InteractionTensor.order_labels

    Returns
    -------
    gini_df : pandas.DataFrame
        Dataframe containing the Gini coefficient of each factor from
        a tensor decomposition. Calculated on the factor-specific
        cell-cell communication networks.
    '''
    if hasattr(result, 'factors'):
        result = result.factors
        if result is None:
            raise ValueError('A tensor factorization must be run on the tensor before calling this function.')
    elif isinstance(result, dict):
        pass
    else:
        raise ValueError('result is not of a valid type. It must be an InteractionTensor or a dict.')

    factors = sorted(list(set(result[sender_label].columns) & set(result[receiver_label].columns)))

    ginis = []
    for f in factors:
        factor_net = get_joint_loadings(result=result,
                                        dim1=sender_label,
                                        dim2=receiver_label,
                                        factor=f
                                        )
        gini = gini_coefficient(factor_net.values.flatten())
        ginis.append((f, gini))
    gini_df = pd.DataFrame.from_records(ginis, columns=['Factor', 'Gini'])
    return gini_df