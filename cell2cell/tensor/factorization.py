# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial as sp
from tqdm.auto import tqdm
from kneed import KneeLocator

from tensorly.decomposition import non_negative_parafac, non_negative_parafac_hals, parafac, constrained_parafac
import tensorly as tl

from cell2cell.tensor.metrics import pairwise_correlation_index


def _compute_tensor_factorization(tensor, rank, tf_type='non_negative_cp', init='svd', svd='numpy_svd', random_state=None,
                                  mask=None, n_iter_max=100, tol=10e-7, verbose=False, **kwargs):
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

        - 'non_negative_cp' : Non-negative PARAFAC through the traditional ALS.
        - 'non_negative_cp_hals' : Non-negative PARAFAC through the Hierarchical ALS.
                                   It reaches an optimal solution faster than the
                                   traditional ALS, but it does not allow a mask.
        - 'parafac' : PARAFAC through the traditional ALS. It allows negative loadings.
        - 'constrained_parafac' : PARAFAC through the traditional ALS. It allows
                                  negative loadings. Also, it incorporates L1 and L2
                                  regularization, includes a 'non_negative' option, and
                                  allows constraining the sparsity of the decomposition.
                                  For more information, see
                                  http://tensorly.org/stable/modules/generated/tensorly.decomposition.constrained_parafac.html#tensorly.decomposition.constrained_parafac

    init : str, default='svd'
        Initialization method for computing the Tensor Factorization.
        {‘svd’, ‘random’}

    svd : str, default='numpy_svd'
        Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS

    random_state : int, default=None
        Seed for randomization.

    mask : ndarray list, default=None
        Helps avoiding missing values during a tensor factorization. A mask should be
        a boolean array of the same shape as the original tensor and should be 0
        where the values are missing and 1 everywhere else.

    tol : float, default=10e-7
        Tolerance for the decomposition algorithm to stop when the variation in
        the reconstruction error is less than the tolerance. Lower `tol` helps
        to improve the solution obtained from the decomposition, but it takes
        longer to run.

    n_iter_max : int, default=100
        Maximum number of iteration to reach an optimal solution with the
        decomposition algorithm. Higher `n_iter_max`helps to improve the solution
        obtained from the decomposition, but it takes longer to run.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    **kwargs : dict
        Extra arguments for the tensor factorization according to inputs in tensorly.

    Returns
    -------
    cp_tf : CPTensor
        A tensorly object containing a list of initialized factors of the tensor
        decomposition where element `i` is of shape (tensor.shape[i], rank).

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
        Returned only when `kwargs` contains 'return_errors' : True.
    '''
    # For returning errors
    return_errors = False
    if kwargs is not None:
        if 'return_errors' in kwargs.keys():
            return_errors = kwargs['return_errors']

    if tf_type == 'non_negative_cp':
        cp_tf = non_negative_parafac(tensor=tensor,
                                     rank=rank,
                                     init='random' if mask is not None else init,
                                     svd=svd,
                                     random_state=random_state,
                                     mask=mask,
                                     n_iter_max=n_iter_max,
                                     tol=tol,
                                     verbose=verbose,
                                     **kwargs)

        if return_errors:
            cp_tf, errors = cp_tf

    elif tf_type == 'non_negative_cp_hals':
        cp_tf, errors = non_negative_parafac_hals(tensor=tensor,
                                                  rank=rank,
                                                  init=init,
                                                  svd=svd,
                                                  #random_state=random_state, # Not implemented in tensorly 0.7.0 commented for now
                                                  n_iter_max=n_iter_max,
                                                  tol=tol,
                                                  verbose=verbose,
                                                  **kwargs)
    elif tf_type == 'parafac':
        cp_tf, errors = parafac(tensor=tensor,
                                rank=rank,
                                init='random' if mask is not None else init,
                                svd=svd,
                                random_state=random_state,
                                mask=mask,
                                n_iter_max=n_iter_max,
                                tol=tol,
                                verbose=verbose,
                                **kwargs)

    elif tf_type == 'constrained_parafac':
        cp_tf, errors = constrained_parafac(tensor=tensor,
                                            rank=rank,
                                            init='random' if mask is not None else init,
                                            svd=svd,
                                            random_state=random_state,
                                            n_iter_max=n_iter_max,
                                            verbose=verbose,
                                            **kwargs)
    else:
        raise ValueError('Not a valid tf_type.')

    if return_errors:
        return cp_tf, errors
    else:
        return cp_tf


def _compute_elbow(loss, S=1.0, curve='convex', direction='decreasing', interp_method='interp1d'):
    '''Computes the elbow of a curve

    Parameters
    ----------
    loss : list
        List of lists or tuples with (x, y) coordinates for a curve.

    S : float, default=1.0
        The sensitivity parameter allows us to adjust how aggressive
        we want Kneedle to be when detecting knees. Smaller values
        for S detect knees quicker, while larger values are more conservative.
        Put simply, S is a measure of how many “flat” points we expect to
        see in the unmodified data curve before declaring a knee.

    curve : str, default='convex'
        If curve='concave', kneed will detect knees. If curve='convex',
        it will detect elbows.

    direction : str, default='decreasing'
        The direction parameter describes the line from left to right on the
        x-axis. If the knee/elbow you are trying to identify is on a positive
        slope use direction='increasing', if the knee/elbow you are trying to
        identify is on a negative slope, use direction='decreasing'.

    interp_method : str, default=''
        This parameter controls the interpolation method for fitting a
        spline to the input x and y data points.
        Valid arguments are 'interp1d' and 'polynomial'.

    Returns
    -------
    rank : int
        X coordinate where the elbow is located in the curve.
    '''
    kneedle = KneeLocator(*zip(*loss), S=S, curve=curve, direction=direction, interp_method=interp_method)
    rank = kneedle.elbow
    return rank


def _run_elbow_analysis(tensor, upper_rank=50, tf_type='non_negative_cp', init='svd', svd='numpy_svd', random_state=None,
                        mask=None, n_iter_max=100, tol=10e-7, verbose=False, disable_pbar=False, **kwargs):
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

    svd : str, default='numpy_svd'
        Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS

    random_state : int, default=None
        Seed for randomization.

    mask : ndarray list, default=None
        Helps avoiding missing values during a tensor factorization. A mask should be
        a boolean array of the same shape as the original tensor and should be 0
        where the values are missing and 1 everywhere else.

    tol : float, default=10e-7
        Tolerance for the decomposition algorithm to stop when the variation in
        the reconstruction error is less than the tolerance. Lower `tol` helps
        to improve the solution obtained from the decomposition, but it takes
        longer to run.

    n_iter_max : int, default=100
        Maximum number of iteration to reach an optimal solution with the
        decomposition algorithm. Higher `n_iter_max`helps to improve the solution
        obtained from the decomposition, but it takes longer to run.

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
                                                          svd=svd,
                                                          random_state=random_state,
                                                          mask=mask,
                                                          n_iter_max=n_iter_max,
                                                          tol=tol,
                                                          verbose=verbose,
                                                          **kwargs)
        # This helps to obtain proper error when the mask is not None.
        if mask is None:
            loss.append((r, tl.to_numpy(errors[-1])))
        else:
            loss.append((r, _compute_norm_error(tensor, tl_object, mask)))

    return loss


def _multiple_runs_elbow_analysis(tensor, upper_rank=50, runs=10, tf_type='non_negative_cp', init='svd', svd='numpy_svd',
                                  metric='error', random_state=None, mask=None, n_iter_max=100, tol=10e-7,
                                  verbose=False, **kwargs):
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

    svd : str, default='numpy_svd'
            Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS

    metric : str, default='error'
        Metric to perform the elbow analysis (y-axis)

        - 'error' : Normalized error to compute the elbow.
        - 'similarity' : Similarity based on CorrIndex (1-CorrIndex).

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
                                                              svd=svd,
                                                              random_state=rs,
                                                              mask=mask,
                                                              n_iter_max=n_iter_max,
                                                              tol=tol,
                                                              verbose=verbose,
                                                              **kwargs)

            if metric == 'error':
                # This helps to obtain proper error when the mask is not None.
                if mask is None:
                    run_errors.append(tl.to_numpy(errors[-1]))
                else:
                    run_errors.append(_compute_norm_error(tensor, tl_object, mask))
            elif metric == 'similarity':
                (weights, factors) = tl_object
                factors = [tl.to_numpy(f) for f in factors]
                run_errors.append(dict(zip(list(range(len(factors))), factors)))

        if metric == 'similarity':
            corridx_df = pairwise_correlation_index(run_errors)
            run_errors = 1.0 - sp.distance.squareform(corridx_df.values)
            #run_errors = 1.0 - corridx_df.max().values
            run_errors = run_errors.tolist()
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
        #diff = tl.tensor(np.ones(mask.shape), **context) - mask_
        #tensor_ = tensor * mask_ + rec_ * diff
        tensor_ = tensor * mask_
        rec_ = rec_ * mask_
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