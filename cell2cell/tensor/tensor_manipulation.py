# -*- coding: utf-8 -*-

import tensorly as tl

from cell2cell.tensor.tensor import PreBuiltTensor
from cell2cell.tensor.subset import subset_tensor


def concatenate_interaction_tensors(interaction_tensors, axis, order_labels, remove_duplicates=False, keep='first',
                                    mask=None, device=None):
    '''Concatenates interaction tensors in a given tensor dimension or axis.

    Parameters
    ----------
    interaction_tensors : list
        List of any tensor class in cell2cell.tensor.

    axis : int
        The axis along which the arrays will be joined. If axis is None, arrays are flattened before use.

    order_labels : list
        List of labels for dimensions or orders in the tensor.

    remove_duplicates : boolean, default=False
        Whether removing duplicated names in the concatenated axis.

    keep : str, default='first'
        Determines which duplicates (if any) to keep.
        Options are:

        - first : Drop duplicates except for the first occurrence.
        - last : Drop duplicates except for the last occurrence.
        - False : Drop all duplicates.

    mask : ndarray list
        Helps avoiding missing values during a tensor factorization. A mask should be
        a boolean array of the same shape as the original tensor and should be 0
        where the values are missing and 1 everywhere else. This must be of equal shape
        as the concatenated tensor.

    device : str, default=None
        Device to use when backend is pytorch. Options are:
         {'cpu', 'cuda', None}

    Returns
    -------
    concatenated_tensor : cell2cell.tensor.PreBuiltTensor
        Final tensor after concatenation. It is a PreBuiltTensor that works
        any interaction tensor based on the class BaseTensor.
    '''
    # Assert if all other dimensions contains the same elements:
    shape = len(interaction_tensors[0].tensor.shape)
    assert all(shape == len(tensor.tensor.shape) for tensor in interaction_tensors[1:]), "Tensors must have same number of dimensions"

    for i in range(shape):
        if i != axis:
            elements = interaction_tensors[0].order_names[i]
            for tensor in interaction_tensors[1:]:
                assert elements == tensor.order_names[i], "Tensors must have the same elements in the other axes."

    # Initialize tensors into a numpy object for performing subset
    # Use the same context as first tensor for everything
    try:
        context = tl.context(interaction_tensors[0].tensor)
    except:
        context = {'dtype': interaction_tensors[0].tensor.dtype, 'device' : None}

    # Concatenate tensors
    concat_tensor = tl.concatenate([tensor.tensor.to('cpu') for tensor in interaction_tensors], axis=axis)
    if mask is not None:
        assert mask.shape == concat_tensor.shape, "Mask must have the same shape of the concatenated tensor. Here: {}".format(concat_tensor.shape)
    else: # Generate a new mask from all previous masks if all are not None
        if all([tensor.mask is not None for tensor in interaction_tensors]):
            mask = tl.concatenate([tensor.mask.to('cpu') for tensor in interaction_tensors], axis=axis)
        else:
            mask = None

    concat_tensor = tl.tensor(concat_tensor, device=context['device'])
    if mask is not None:
        mask = tl.tensor(mask, device=context['device'])

    # Concatenate names of elements for the given axis but keep the others as in one tensor
    order_names = []
    for i in range(shape):
        tmp_names = []
        if i == axis:
            for tensor in interaction_tensors:
                tmp_names += tensor.order_names[i]
        else:
            tmp_names = interaction_tensors[0].order_names[i]
        order_names.append(tmp_names)

    # Generate final object
    concatenated_tensor = PreBuiltTensor(tensor=concat_tensor,
                                         order_names=order_names,
                                         order_labels=order_labels,
                                         mask=mask,  # Change if you want to omit values in the decomposition
                                         device=device
                                        )

    # Remove duplicates
    if remove_duplicates:
        concatenated_tensor = subset_tensor(interaction_tensor=concatenated_tensor,
                                            subset_dict={axis: order_names[axis]},
                                            remove_duplicates=remove_duplicates,
                                            keep=keep,
                                            original_order=False)
    return concatenated_tensor