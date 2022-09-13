# -*- coding: utf-8 -*-
import copy

import numpy as np
import tensorly as tl

from cell2cell.preprocessing.find_elements import find_duplicates

def find_element_indexes(interaction_tensor, elements, axis=0, remove_duplicates=True, keep='first', original_order=False):
    '''Finds the location/indexes of a list of elements in one of the
    axis of an InteractionTensor.

    Parameters
    ----------
    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor generated with any of the tensor class in
        cell2cell.tensor

    elements : list
        A list of names for the elements to find in one of the axis.

    axis : int, default=0
        An axis of the interaction_tensor, representing one of
        its dimensions.

    remove_duplicates : boolean, default=True
        Whether removing duplicated names in `elements`.

    keep : str, default='first'
        Determines which duplicates (if any) to keep.
        Options are:

        - first : Drop duplicates except for the first occurrence.
        - last : Drop duplicates except for the last occurrence.
        - False : Drop all duplicates.

    original_order : boolean, default=False
        Whether keeping the original order of the elements in
        interaction_tensor.order_names[axis] or keeping the
        new order as indicated in `elements`.

    Returns
    -------
    indexes : list
        List of indexes for the elements that where found in the
        axis indicated of the interaction_tensor.
    '''
    assert axis < len \
        (interaction_tensor.tensor.shape), "List index out of range. 'axis' must be one of the axis in the tensor."
    assert axis < len \
        (interaction_tensor.order_names), "List index out of range. interaction_tensor.order_names must have element names for each axis of the tensor."

    elements = sorted(set(elements), key=list(elements).index)

    if original_order:
        # Avoids error for considering elements not in the tensor
        elements = set(elements).intersection(set(interaction_tensor.order_names[axis]))
        elements = sorted(elements, key=interaction_tensor.order_names[axis].index)


    # Find duplicates if we are removing them
    to_exclude = []
    if remove_duplicates:
        dup_dict = find_duplicates(interaction_tensor.order_names[axis])

        if len(dup_dict) > 0:  # Only if we have duplicate items
            if keep == 'first':
                for k, v in dup_dict.items():
                    to_exclude.extend(v[1:])
            elif keep == 'last':
                for k, v in dup_dict.items():
                    to_exclude.extend(v[:-1])
            elif not keep:
                for k, v in dup_dict.items():
                    to_exclude.extend(v)
            else:
                raise ValueError("Not a valid option was selected for the parameter `keep`")

    # Find indexes in the tensor
    indexes = sum \
        ([np.where(np.asarray(interaction_tensor.order_names[axis]) == element)[0].tolist() for element in elements], [])

    # Exclude duplicates if any to exclude
    indexes = [idx for idx in indexes if idx not in to_exclude]
    return indexes


def subset_tensor(interaction_tensor, subset_dict, remove_duplicates=True, keep='first', original_order=False):
    '''Subsets an InteractionTensor to contain only specific elements in
    respective dimensions.

    Parameters
    ----------
    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor generated with any of the tensor class in
        cell2cell.tensor

    subset_dict : dict
        Dictionary to subset the tensor. It must contain the axes or
        dimensions that will be subset as the keys of the dictionary
        and the values corresponds to lists of element names for the
        respective axes or dimensions. Those axes that are not present
        in this dictionary will not be subset.
        E.g. {0 : ['Context 1', 'Context2'], 1: ['LR 10', 'LR 100']}

    remove_duplicates : boolean, default=True
        Whether removing duplicated names in `elements`.

    keep : str, default='first'
        Determines which duplicates (if any) to keep.
        Options are:

        - first : Drop duplicates except for the first occurrence.
        - last : Drop duplicates except for the last occurrence.
        - False : Drop all duplicates.

    original_order : boolean, default=False
        Whether keeping the original order of the elements in
        interaction_tensor.order_names or keeping the
        new order as indicated in the lists in the `subset_dict`.

    Returns
    -------
    subset_tensor : cell2cell.tensor.BaseTensor
        A copy of interaction_tensor that was subset to contain
        only the elements specified for the respective axis in the
        `subset_dict`. Corresponds to a communication tensor
        generated with any of the tensor class in cell2cell.tensor
    '''
    # Perform a deep copy of the original tensor and reset previous factorization
    subset_tensor = copy.deepcopy(interaction_tensor)
    subset_tensor.rank = None
    subset_tensor.tl_object = None
    subset_tensor.factors = None

    # Initialize tensor into a numpy object for performing subset
    context = tl.context(subset_tensor.tensor)
    tensor = tl.to_numpy(subset_tensor.tensor)
    mask = None
    if subset_tensor.mask is not None:
        mask = tl.to_numpy(subset_tensor.mask)

    # Search for indexes
    axis_idxs = dict()
    for k, v in subset_dict.items():
        if k < len(tensor.shape):
            if len(v) != 0:
                idx = find_element_indexes(interaction_tensor=subset_tensor,
                                           elements=v,
                                           axis=k,
                                           remove_duplicates=remove_duplicates,
                                           keep=keep,
                                           original_order=original_order
                                           )
                if len(idx) == 0:
                    print("No elements found for axis {}. It will return an empty tensor.".format(k))
                axis_idxs[k] = idx
        else:
            print("Axis {} is out of index, not considering elements in this axis.".format(k))

    # Subset tensor
    for k, v in axis_idxs.items():
        if tensor.shape != (0,):  # Avoids error when returned empty tensor
            tensor = tensor.take(indices=v,
                                 axis=k
                                 )

            subset_tensor.order_names[k] = [subset_tensor.order_names[k][i] for i in v]
            if mask is not None:
                mask = mask.take(indices=v,
                                 axis=k
                                 )

    # Restore tensor and mask properties
    tensor = tl.tensor(tensor, **context)
    if mask is not None:
        mask = tl.tensor(mask, **context)

    subset_tensor.tensor = tensor
    subset_tensor.mask = mask
    return subset_tensor


def subset_metadata(tensor_metadata, interaction_tensor, sample_col='Element'):
    '''Subsets the metadata of an InteractionTensor to contain only
    elements in a reference InteractionTensor (interaction_tensor).

    Parameters
    ----------
    tensor_metadata : list
        List of pandas dataframes with metadata information for elements of each
        dimension in the tensor. A column called as the variable `sample_col` contains
        the name of each element in the tensor while another column called as the
        variable `group_col` contains the metadata or grouping information of each
        element.

    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor generated with any of the tensor class in
        cell2cell.tensor. This tensor is used as reference to subset the metadata.
        The subset metadata will contain only elements that are present in this
        tensor, so if metadata was originally built for another tensor, the elements
        that are exclusive for that original tensor will be excluded.

    sample_col : str, default='Element'
        Name of the column containing the element names in the metadata.

    Returns
    -------
    subset_metadata : list
        List of pandas dataframes with metadata information for elements contained
        in `interaction_tensor.order_names`. It is a subset of `tensor_metadata`.
    '''
    subset_metadata = []
    for i, meta in enumerate(tensor_metadata):
        if meta is not None:
            tmp_meta = meta.set_index(sample_col)
            tmp_meta = tmp_meta.loc[interaction_tensor.order_names[i], :]
            tmp_meta = tmp_meta.reset_index()
            subset_metadata.append(tmp_meta)
        else:
            subset_metadata.append(None)
    return subset_metadata