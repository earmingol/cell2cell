# -*- coding: utf-8 -*-

import numpy as np
from tqdm.auto import tqdm
from kneed import KneeLocator

from cell2cell.external import non_negative_parafac
# Replace previous line with line below once the random_state is available for init='random'
#from tensorly.decomposition import non_negative_parafac
import tensorly as tl


def _compute_tensor_factorization(tensor, rank, tf_type='non_negative_cp', init='svd', random_state=None, mask=None,
                                  verbose=False, **kwargs):
    '''Performs the Tensor Factorization

    Parameters
    ---------
    tensor : ndarray list
        A tensor that could be a list of lists, a multidimensional numpy array or a
        tensorly.tensor.

    rank : int
        Rank of the Tensor Factorization (number of factors to deconvolve the original
        tensor).

    tf_type : str, default='non_negative_cp'
        Type of Tensor Factorization.
        - 'non_negative_cp' : Non-negative PARAFAC, as implemented in Tensorly

    init : str, default='svd'
        Initialization method for computing the Tensor Factorization.
        {‘svd’, ‘random’}

    random_state : int, default=None
        Seed for randomization.

    mask : ndarray list, default=None
        Helps avoiding missing values during a tensor factorization. A mask should be
        a boolean array of the same shape as the original tensor and should be 0
        where the values are missing and 1 everywhere else.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    **kwargs : dict
        Extra arguments for the tensor factorization according to inputs in tensorly.

    Returns
    -------
    cp_tf : CPTensor
        A tensorly object containing a list of initialized factors of the tensor
        decomposition where element `i` is of shape (tensor.shape[i], rank).
    '''
    if mask is not None:
        init = 'random'
    if tf_type == 'non_negative_cp':
        cp_tf = non_negative_parafac(tensor=tensor,
                                     rank=rank,
                                     init=init,
                                     random_state=random_state,
                                     mask=mask,
                                     verbose=verbose,
                                     **kwargs)
    else:
        raise ValueError('Not a valid tf_type.')

    return cp_tf


def _compute_elbow(loss):
    '''Computes the elbow of a curve

    Parameters
    ----------
    loss : list
        List of lists or tuples with (x, y) coordinates for a curve.

    Returns
    -------
    rank : int
        X coordinate where the elbow is located in the curve.
    '''
    kneedle = KneeLocator(*zip(*loss), S=1.0, curve="convex", direction="decreasing")
    rank = kneedle.elbow
    return rank


def _run_elbow_analysis(tensor, upper_rank=50, tf_type='non_negative_cp', init='svd', random_state=None, mask=None,
                        verbose=False, disable_pbar=False, **kwargs):
    '''Performs an elbow analysis with just one run of a tensor factorization for each
    rank

    Parameters
    ----------
    tensor : ndarray list
        A tensor that could be a list of lists, a multidimensional numpy array or a
        tensorly.tensor.

    upper_rank : int, default=50
        Upper bound of ranks to explore with the elbow analysis.

    tf_type : str, default='non_negative_cp'
        Type of Tensor Factorization.
        - 'non_negative_cp' : Non-negative PARAFAC, as implemented in Tensorly

    init : str, default='svd'
        Initialization method for computing the Tensor Factorization.
        {‘svd’, ‘random’}

    random_state : int, default=None
        Seed for randomization.

    mask : ndarray list, default=None
        Helps avoiding missing values during a tensor factorization. A mask should be
        a boolean array of the same shape as the original tensor and should be 0
        where the values are missing and 1 everywhere else.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    disable_pbar : boolean, default=False
        Whether displaying a tqdm progress bar or not.

    **kwargs : dict
        Extra arguments for the tensor factorization according to inputs in tensorly.

    Returns
    -------
    loss : list
        List of  tuples with (x, y) coordinates for the elbow analysis. X values are
        the different ranks and Y values are the errors of each decomposition.
    '''
    if kwargs is None:
        kwargs = {'return_errors': True}
    else:
        kwargs['return_errors'] = True

    loss = []
    for r in tqdm(range(1, upper_rank + 1), disable=disable_pbar):
        tl_object, errors = _compute_tensor_factorization(tensor=tensor,
                                                          rank=r,
                                                          tf_type=tf_type,
                                                          init=init,
                                                          random_state=random_state,
                                                          mask=mask,
                                                          verbose=verbose,
                                                          **kwargs)
        # This helps to obtain proper error when the mask is not None.
        if mask is None:
            loss.append((r, errors[-1]))
        else:
            loss.append((r, _compute_norm_error(tensor, tl_object, mask)))

    return loss


def _multiple_runs_elbow_analysis(tensor, upper_rank=50, runs=10, tf_type='non_negative_cp', init='svd', random_state=None,
                                  mask=None, verbose=False, **kwargs):
    '''Performs an elbow analysis with multiple runs of a tensor factorization for each
    rank

    Parameters
    ----------
    tensor : ndarray list
        A tensor that could be a list of lists, a multidimensional numpy array or a
        tensorly.tensor.

    upper_rank : int, default=50
        Upper bound of ranks to explore with the elbow analysis.

    runs : int, default=100
        Number of tensor factorization performed for a given rank. Each factorization
        varies in the seed of initialization.

    tf_type : str, default='non_negative_cp'
        Type of Tensor Factorization.
        - 'non_negative_cp' : Non-negative PARAFAC, as implemented in Tensorly

    init : str, default='svd'
        Initialization method for computing the Tensor Factorization.
        {‘svd’, ‘random’}

    random_state : int, default=None
        Seed for randomization.

    mask : ndarray list, default=None
        Helps avoiding missing values during a tensor factorization. A mask should be
        a boolean array of the same shape as the original tensor and should be 0
        where the values are missing and 1 everywhere else.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    **kwargs : dict
        Extra arguments for the tensor factorization according to inputs in tensorly.

    Returns
    -------
    all_loss : ndarray
        Array containing the errors associated with multiple runs for a given rank.
        This array is of shape (runs, upper_rank).
    '''
    assert isinstance(runs, int), "runs must be an integer"
    if kwargs is None:
        kwargs = {'return_errors': True}
    else:
        kwargs['return_errors'] = True

    all_loss = []
    for r in tqdm(range(1, upper_rank + 1)):
        run_errors = []
        for run in range(runs):
            if random_state is not None:
                rs = random_state + run
            else:
                rs = None
            tl_object, errors = _compute_tensor_factorization(tensor=tensor,
                                                              rank=r,
                                                              tf_type=tf_type,
                                                              init=init,
                                                              random_state=rs,
                                                              mask=mask,
                                                              verbose=verbose,
                                                              **kwargs)
            # This helps to obtain proper error when the mask is not None.
            if mask is None:
                run_errors.append(errors[-1])
            else:
                run_errors.append(_compute_norm_error(tensor, tl_object, mask))
        all_loss.append(run_errors)

    all_loss = np.array(all_loss).T
    return all_loss


def _compute_norm_error(tensor, tl_object, mask=None):
    '''Computes the normalized error of a tensor factorization

    The regular error is : ||tensor - reconstructed_tensor||^2;
    Where  ||.||^2 represent the squared Frobenius. A reconstructed_tensor is computed
    from transforming the original tensor with the factors of the decomposition.
    Thus, the normalized error is: ||tensor - reconstructed_tensor||^2 / ||tensor||^2;
    which has values between 0 and 1.

    Parameters
    ----------
    tensor : ndarray list
        A tensor that could be a list of lists, a multidimensional numpy array or a
        tensorly.tensor.

    tl_object : CPTensor
        A tensorly object containing a list of initialized factors of the tensor
        decomposition where element `i` is of shape (tensor.shape[i], rank).

    mask : ndarray list, default=None
        Helps avoiding missing values during a tensor factorization. A mask should be
        a boolean array of the same shape as the original tensor and should be 0
        where the values are missing and 1 everywhere else.

    Returns
    -------
    l : float
        The normalized error of a tensor decomposition.
    '''
    assert tl_object is not None, "Before computing the error, please run the function _compute_tensor_factorization to generate the tl_object"

    # Compute error
    rec_ = tl_object.to_tensor()
    context = tl.context(tensor)
    if mask is not None:
        if context != tl.context(mask):
            mask_ = tl.tensor(mask, **context)
        else:
            mask_ = mask
        diff = tl.tensor(np.ones(mask.shape), **context) - mask_
        tensor_ = tensor * mask_ + rec_ * diff
    else:
        tensor_ = tensor

    # error = ||tensor - rec||^2 = ||tensor||^2 + ||rec||^2 - 2*<tensor, rec>
    #iprod = np.nansum(np.inner(tensor_, rec_))
    #norm_tensor = np.linalg.norm(tensor_)
    #norm_rec = np.linalg.norm(rec_)
    #error = np.sqrt(np.abs(norm_tensor ** 2 + norm_rec ** 2 - 2 * iprod))

    # Normalized error
    #l = error / norm_tensor
    l = normalized_error(reference_tensor=tensor_,
                         reconstructed_tensor=rec_)
    return l


def normalized_error(reference_tensor, reconstructed_tensor):
    '''Computes a normalized error between two tensors

    Parameters
    ----------
    reference_tensor : ndarray list
        A tensor that could be a list of lists, a multidimensional numpy array or
        a tensorly.tensor. This tensor is the input of a tensor decomposition and
        used as reference in the normalized error for a new tensor reconstructed
        from the factors of the tensor decomposition.

    reconstructed_tensor : ndarray list
        A tensor that could be a list of lists, a multidimensional numpy array or
        a tensorly.tensor. This tensor is an approximation of the reference_tensor
        by using the resulting factors of a tensor decomposition to compute it.

    Returns
    -------
    norm_error : float
        The normalized error between a reference tensor and a reconstructed tensor.
        The error is normalized by dividing by the Frobinius norm of the reference
        tensor.
    '''
    norm_error = tl.norm(reference_tensor - reconstructed_tensor) / tl.norm(reference_tensor)
    return tl.to_numpy(norm_error)