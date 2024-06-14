# -*- coding: utf-8 -*-
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from tensorly.tenalg import outer

from cell2cell.stats import gini_coefficient
from cell2cell.tensor.metrics import error_dict

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

    assert dim1 in result.keys(), 'The specified dimension ' + dim1 + ' is not present in the `result` input'
    assert dim2 in result.keys(), 'The specified dimension ' + dim2 + ' is not present in the `result` input'

    vec1 = result[dim1][factor]
    vec2 = result[dim2][factor]

    # Calculate the outer product
    joint_dist = pd.DataFrame(data=np.outer(vec1, vec2),
                              index=vec1.index,
                              columns=vec2.index)

    joint_dist.index.name = dim1
    joint_dist.columns.name = dim2
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


def get_lr_by_cell_pairs(result, lr_label, sender_label, receiver_label, order_cells_by='receivers', factor=None,
                         cci_threshold=None, lr_threshold=None):
    '''
    Returns a dataframe containing the product loadings of a specific combination
    of ligand-receptor pair and sender-receiver pair.

    Parameters
    ----------
    result : any Tensor class in cell2cell.tensor.tensor or a dict
        Either a Tensor type or a dictionary which resulted from the tensor
        decomposition. If it is a dict, it should be the one in, for example,
        InteractionTensor.factors

    lr_label : str
        Label for the dimension of the ligand-receptor pairs. Usually found in
        InteractionTensor.order_labels

    sender_label : str
        Label for the dimension of sender cells. Usually found in
        InteractionTensor.order_labels

    receiver_label : str
        Label for the dimension of receiver cells. Usually found in
        InteractionTensor.order_labels

    order_cells_by : str, default='receivers'
        Order of the returned dataframe. Options are 'senders' and
        'receivers'. 'senders' means to order the dataframe in a way that
        all cell-cell pairs with a same sender cell are put next to each others.
        'receivers' means the same, but by considering the receiver cell instead.

    factor : str, default=None
        Name of the factor to be used to compute the product loadings.
        If None, all factors will be included to compute them.

    cci_threshold : float, default=None
        Threshold to be applied on the product loadings of the sender-cell pairs.
        If specified, only cell-cell pairs with a product loading above the
        threshold at least in one of the factors included will be included
        in the returned dataframe.

    lr_threshold : float, default=None
        Threshold to be applied on the ligand-receptor loadings.
        If specified, only LR pairs with a loading above the
        threshold at least in one of the factors included will be included
        in the returned dataframe.

    Returns
    -------
    cci_lr : pandas.DataFrame
        Dataframe containing the product loadings of a specific combination 
        of ligand-receptor pair and sender-receiver pair. If the factor is specified,
        the returned dataframe will contain the product loadings of that factor.
        If the factor is not specified, the returned dataframe will contain the
        product loadings across all factors.
    '''
    if hasattr(result, 'factors'):
        result = result.factors
        if result is None:
            raise ValueError('A tensor factorization must be run on the tensor before calling this function.')
    elif isinstance(result, dict):
        pass
    else:
        raise ValueError('result is not of a valid type. It must be an InteractionTensor or a dict.')

    assert lr_label in result.keys(), 'The specified dimension ' + lr_label + ' is not present in the `result` input'
    assert sender_label in result.keys(), 'The specified dimension ' + sender_label + ' is not present in the `result` input'
    assert receiver_label in result.keys(), 'The specified dimension ' + receiver_label + ' is not present in the `result` input'

    # Sort factors
    sorted_factors = sorted(result[lr_label].columns, key=lambda x: int(x.split(' ')[1]))

    # Get CCI network per factor
    networks = get_factor_specific_ccc_networks(result=result,
                                                sender_label=sender_label,
                                                receiver_label=receiver_label)

    # Flatten networks
    network_by_factors = flatten_factor_ccc_networks(networks=networks, orderby=order_cells_by)

    # Get final dataframe
    df1 = network_by_factors[sorted_factors]
    df2 = result[lr_label][sorted_factors]

    if factor is not None:
        df1 = df1[factor]
        df2 = df2[factor]
        if cci_threshold is not None:
            df1 = df1[(df1 > cci_threshold)]
        if lr_threshold is not None:
            df2 = df2[(df2 > lr_threshold)]
        data = pd.DataFrame(np.outer(df1, df2), index=df1.index, columns=df2.index)
    else:
        if cci_threshold is not None:
            df1 = df1[(df1.T > cci_threshold).any()]  # Top sender-receiver pairs
        if lr_threshold is not None:
            df2 = df2[(df2.T > lr_threshold).any()]  # Top LR Pairs
        data = np.matmul(df1, df2.T)

    cci_lr = pd.DataFrame(data.T.values,
                          columns=df1.index,
                          index=df2.index
                          )

    cci_lr.columns.name = 'Sender-Receiver Pair'
    cci_lr.index.name = 'Ligand-Receptor Pair'
    return cci_lr

def get_pairwise_factor_distance(tensor,#: cell2cell.tensor.tensor.InteractionTensor, 
                                 factors: Optional[List[str]] = None,
                                metric: Literal['frobenius', 'rmse', 'mae'] = 'frobenius') -> pd.DataFrame:
    """Computes the pairwise distance between the rank-1 tensors of each factor

    Example code: `pf = get_pairwise_factor_distance(tensor, factors = ['Factor 1', 'Factor 3'], metric = 'rmse')`

    Parameters
    ----------
    tensor : cell2cell.tensor.tensor.InteractionTensor
        input tensor with decomposition loadings stored in the `factors` attribute
    factors : Optional[List[Tuple[str]]], optional
        the subset of factors to compare, by default None
    metric : Literal['frobenius', 'rmse', 'mae'], optional
        the distance metric to use, by default 'frobenius'


    Returns
    -------
    pd.DataFrame
        pairwise distances between factors
    """
    # if not factors: # get all pairwise comparisons
    #     all_factors = list(tensor.factors.values())[0].columns
    #     factors = list(itertools.combinations(all_factors, 2))
    # unique_factors = sorted(set([item for tup in factors for item in tup]))
    if not factors:
        factors = list(tensor.factors.values())[0].columns
    
    factor_tensors = {}
    for factor in factors:
        factor_tensor = []
        for key in tensor.factors.keys():
            factor_tensor.append(tensor.factors[key].loc[:, factor].values)
        factor_tensors[factor] = outer(factor_tensor)
    
    pf = error_dict[metric](list(factor_tensors.values()))
    pf = pd.DataFrame(pf, columns = factors, index = factors)
    return pf