### This was borrowed temporarily from the version 0.5.1 of tensorly: https://github.com/tensorly/tensorly/tree/main/tensorly
### https://pypi.org/project/tensorly/0.5.1/#files
### Once the non_negative_parafac takes random_state when init='random', this will replaces by the corresponding version.
### TO REMEMBER: When removing this, remove version requisite of tensorly in the setup.py file

import warnings
import tensorly as tl
from tensorly.random import random_cp # check_random_state, # check_random_state only available in tensorly 0.5.1
from tensorly.base import unfold
from tensorly.cp_tensor import (CPTensor,
                         unfolding_dot_khatri_rao, cp_norm,
                         cp_normalize, validate_cp_rank)


def initialize_cp(tensor, rank, init='svd', svd='numpy_svd', random_state=None,
                       non_negative=False, normalize_factors=False):
    r"""Initialize factors used in `parafac`.
    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor.
    Parameters
    ----------
    tensor : ndarray
    rank : int
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    non_negative : bool, default is False
        if True, non-negative factors are returned
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    rng = check_random_state(random_state)

    if init == 'random':
        # factors = [tl.tensor(rng.random_sample((tensor.shape[i], rank)), **tl.context(tensor)) for i in range(tl.ndim(tensor))]
        # kt = CPTensor((None, factors))
        return random_cp(tl.shape(tensor), rank, normalise_factors=False, random_state=rng, **tl.context(tensor))

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                    svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, S, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank)

            # Put SVD initialization on the same scaling as the tensor in case normalize_factors=False
            if mode == 0:
                idx = min(rank, tl.shape(S)[0])
                U = tl.index_update(U, tl.index[:, :idx], U[:, :idx] * S[:idx])

            if tensor.shape[mode] < rank:
                # TODO: this is a hack but it seems to do the job for now
                # factor = tl.tensor(np.zeros((U.shape[0], rank)), **tl.context(tensor))
                # factor[:, tensor.shape[mode]:] = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                # factor[:, :tensor.shape[mode]] = U
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)

            factors.append(U[:, :rank])

        kt = CPTensor((None, factors))

    elif isinstance(init, (tuple, list, CPTensor)):
        # TODO: Test this
        try:
            kt = CPTensor(init)
        except ValueError:
            raise ValueError(
                'If initialization method is a mapping, then it must '
                'be possible to convert it to a CPTensor instance'
            )
    else:
        raise ValueError('Initialization method "{}" not recognized'.format(init))

    if non_negative:
        kt.factors = [tl.abs(f) for f in kt[1]]

    if normalize_factors:
        kt = cp_normalize(kt)

    return kt


def non_negative_parafac(tensor, rank, n_iter_max=100, init='svd', svd='numpy_svd',
                         tol=10e-7, random_state=None, verbose=0, normalize_factors=False,
                         return_errors=False, mask=None, orthogonalise=False, cvg_criterion='abs_rec_error',
                         fixed_modes=[]):
    """
    Non-negative CP decomposition
    Uses multiplicative updates, see [2]_
    This is the same as parafac(non_negative=True).
    Parameters
    ----------
    tensor : ndarray
    rank   : int
            number of components
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity
    fixed_modes : list, default is []
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.
    Returns
    -------
    factors : ndarray list
            list of positive factors of the CP decomposition
            element `i` is of shape ``(tensor.shape[i], rank)``
    References
    ----------
    .. [2] Amnon Shashua and Tamir Hazan,
       "Non-negative tensor factorization with applications to statistics and computer vision",
       In Proceedings of the International Conference on Machine Learning (ICML),
       pp 792-799, ICML, 2005
    """
    epsilon = 10e-12
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)

    if mask is not None and init == "svd":
        message = "Masking occurs after initialization. Therefore, random initialization is recommended."
        warnings.warn(message, Warning)

    if orthogonalise and not isinstance(orthogonalise, int):
        orthogonalise = n_iter_max

    weights, factors = initialize_cp(tensor, rank, init=init, svd=svd,
                                     random_state=random_state,
                                     non_negative=True,
                                     normalize_factors=normalize_factors)
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    if tl.ndim(tensor) - 1 in fixed_modes:
        warnings.warn(
            'You asked for fixing the last mode, which is not supported while tol is fixed.\n The last mode will not be fixed. Consider using tl.moveaxis()')
        fixed_modes.remove(tl.ndim(tensor) - 1)
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    for iteration in range(n_iter_max):
        if orthogonalise and iteration <= orthogonalise:
            for i, f in enumerate(factors):
                if min(tl.shape(f)) >= rank:
                    factors[i] = tl.abs(tl.qr(f)[0])

        if verbose > 1:
            print("Starting iteration", iteration + 1)
        for mode in modes_list:
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))

            accum = 1
            # khatri_rao(factors).tl.dot(khatri_rao(factors))
            # simplifies to multiplications
            sub_indices = [i for i in range(len(factors)) if i != mode]
            for i, e in enumerate(sub_indices):
                if i:
                    accum *= tl.dot(tl.transpose(factors[e]), factors[e])
                else:
                    accum = tl.dot(tl.transpose(factors[e]), factors[e])

            if mask is not None:
                tensor = tensor * mask + tl.cp_to_tensor((None, factors), mask=1 - mask)

            mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

            numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
            denominator = tl.dot(factors[mode], accum)
            denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
            factor = factors[mode] * numerator / denominator

            factors[mode] = factor

        if normalize_factors:
            weights, factors = cp_normalize((weights, factors))

        if tol:
            # ||tensor - rec||^2 = ||tensor||^2 + ||rec||^2 - 2*<tensor, rec>
            factors_norm = cp_norm((weights, factors))

            # mttkrp and factor for the last mode. This is equivalent to the
            # inner product <tensor, factorization>
            iprod = tl.sum(tl.sum(mttkrp * factor, axis=0) * weights)
            rec_error = tl.sqrt(tl.abs(norm_tensor ** 2 + factors_norm ** 2 - 2 * iprod)) / norm_tensor
            rec_errors.append(rec_error)
            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print("iteration {}, reconstraction error: {}, decrease = {}".format(iteration, rec_error,
                                                                                         rec_error_decrease))

                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print("PARAFAC converged after {} iterations".format(iteration))
                    break
            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    cp_tensor = CPTensor((weights, factors))

    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor