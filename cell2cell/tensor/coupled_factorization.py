# -*- coding: utf-8 -*-
# This code is an adaptation built upon the non_negative_parafac function in tensorly.
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp, error_calc
from tensorly.cp_tensor import (
    CPTensor,
    unfolding_dot_khatri_rao,
    cp_normalize,
    validate_cp_rank,
)


def coupled_non_negative_parafac(
        tensor1,
        tensor2,
        rank,
        non_shared_mode,
        mask1=None,
        mask2=None,
        n_iter_max=100,
        init="svd",
        svd="truncated_svd",
        tol=10e-7,
        random_state=None,
        verbose=0,
        normalize_factors=False,
        return_errors=False,
        cvg_criterion="abs_rec_error",
        balance_errors=True,
):
    """
    Coupled Non-negative CP decomposition for two tensors sharing all but one mode

    Performs simultaneous non-negative CP decomposition on two tensors that share
    all dimensions except one. The shared dimensions will have the same factor
    matrices, while the non-shared dimension will have separate factor matrices.

    The convergence can be balanced to ensure equal contribution from both tensors
    regardless of their size differences in the non-shared dimension.

    Parameters
    ----------
    tensor1 : ndarray
        First input tensor
    tensor2 : ndarray
        Second input tensor
    rank : int
        Number of components
    non_shared_mode : int
        The mode (dimension) that differs between the two tensors
    mask1 : ndarray, optional
        Mask for tensor1, if provided, will be used to compute the reconstruction error
    mask2 : ndarray, optional
        Mask for tensor2, if provided, will be used to compute the reconstruction error
    n_iter_max : int, default=100
        Maximum number of iterations
    init : {'svd', 'random'}, optional
        Initialization method
    svd : str, default='truncated_svd'
        SVD function to use
    tol : float, optional, default=1e-7
        Convergence tolerance
    random_state : {None, int, np.random.RandomState}
        Random state for reproducibility
    verbose : int, optional
        Verbosity level
    normalize_factors : bool, default=False
        Whether to normalize factors
    return_errors : bool, default=False
        Whether to return reconstruction errors
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
        Convergence criterion
    balance_errors : bool, default=True
        Whether to balance the errors based on the size of non-shared dimensions.
        If True, errors are weighted by (N1+N2)/N1 and (N1+N2)/N2 where N1, N2
        are the sizes of the non-shared dimensions

    Returns
    -------
    cp_tensor1 : CPTensor
        CP decomposition of tensor1
    cp_tensor2 : CPTensor
        CP decomposition of tensor2
    errors : list (optional)
        Reconstruction errors if return_errors=True

    Examples
    --------
    >>> # Two tensors sharing dimensions 0,1,2 but different in dimension 3
    >>> tensor1 = tl.random.random((10, 20, 30, 40))
    >>> tensor2 = tl.random.random((10, 20, 30, 50))
    >>> cp1, cp2 = coupled_non_negative_parafac(tensor1, tensor2, rank=5, non_shared_mode=3)
    """

    epsilon = tl.eps(tensor1.dtype)

    # Validate inputs
    if tl.ndim(tensor1) != tl.ndim(tensor2):
        raise ValueError("Both tensors must have the same number of dimensions")

    n_modes = tl.ndim(tensor1)
    if non_shared_mode >= n_modes or non_shared_mode < 0:
        raise ValueError(f"non_shared_mode must be between 0 and {n_modes - 1}")

    # Check that all other dimensions match
    for mode in range(n_modes):
        if mode != non_shared_mode and tensor1.shape[mode] != tensor2.shape[mode]:
            raise ValueError(f"Tensors must have the same size in mode {mode}")

    # Validate rank
    rank = validate_cp_rank(tl.shape(tensor1), rank=rank)

    # Calculate sizes of non-shared dimensions for error balancing
    N1 = tensor1.shape[non_shared_mode]
    N2 = tensor2.shape[non_shared_mode]

    if balance_errors:
        # Calculate weights to balance the contribution of each tensor
        total_N = N1 + N2
        weight1 = total_N / N1
        weight2 = total_N / N2
    else:
        # Equal weights if not balancing
        weight1 = 1.0
        weight2 = 1.0

    # Initialize factors for both tensors
    weights1, factors1 = initialize_cp(
        tensor1, rank, mask=mask1, init=init, svd=svd, non_negative=True,
        random_state=random_state, normalize_factors=normalize_factors
    )

    weights2, factors2 = initialize_cp(
        tensor2, rank, mask=mask2, init=init, svd=svd, non_negative=True,
        random_state=random_state, normalize_factors=normalize_factors
    )

    # Use shared weights (average of both initializations)
    weights = (weights1 + weights2) / 2

    # Copy shared factors from tensor1 initialization (could also average)
    for mode in range(n_modes):
        if mode != non_shared_mode:
            factors2[mode] = tl.copy(factors1[mode])

    # Store norms for convergence checking
    norm_tensor1 = tl.norm(tensor1, 2)
    norm_tensor2 = tl.norm(tensor2, 2)

    rec_errors1 = []
    rec_errors2 = []

    for iteration in range(n_iter_max):
        if verbose > 1:
            print(f"Starting iteration {iteration + 1}")

        # Update each mode
        for mode in range(n_modes):
            if verbose > 1:
                print(f"Mode {mode} of {n_modes}")

            if mode == non_shared_mode:
                # Update non-shared mode separately for each tensor

                # Update for tensor1
                accum1 = tl.ones((rank, rank), **tl.context(tensor1))
                for i in range(n_modes):
                    if i != mode:
                        accum1 *= tl.dot(tl.transpose(factors1[i]), factors1[i])
                accum1 = tl.reshape(weights, (-1, 1)) * accum1 * tl.reshape(weights, (1, -1))
                if mask1 is not None:
                    tensor1 = tensor1 * mask1 + tl.cp_to_tensor(
                        (weights, factors1), mask=1 - mask1
                    )

                mttkrp1 = unfolding_dot_khatri_rao(tensor1, (weights, factors1), mode)
                numerator1 = tl.clip(mttkrp1, a_min=epsilon, a_max=None)
                denominator1 = tl.clip(tl.dot(factors1[mode], accum1), a_min=epsilon, a_max=None)
                factors1[mode] = factors1[mode] * numerator1 / denominator1

                # Update for tensor2
                accum2 = tl.ones((rank, rank), **tl.context(tensor2))
                for i in range(n_modes):
                    if i != mode:
                        accum2 *= tl.dot(tl.transpose(factors2[i]), factors2[i])
                accum2 = tl.reshape(weights, (-1, 1)) * accum2 * tl.reshape(weights, (1, -1))
                if mask2 is not None:
                    tensor2 = tensor2 * mask2 + tl.cp_to_tensor(
                        (weights, factors2), mask=1 - mask2
                    )

                mttkrp2 = unfolding_dot_khatri_rao(tensor2, (weights, factors2), mode)
                numerator2 = tl.clip(mttkrp2, a_min=epsilon, a_max=None)
                denominator2 = tl.clip(tl.dot(factors2[mode], accum2), a_min=epsilon, a_max=None)
                factors2[mode] = factors2[mode] * numerator2 / denominator2

            else:
                # Update shared modes using combined information from both tensors

                # Compute accumulation matrix for tensor1
                accum1 = tl.ones((rank, rank), **tl.context(tensor1))
                for i in range(n_modes):
                    if i != mode:
                        if i == non_shared_mode:
                            accum1 *= tl.dot(tl.transpose(factors1[i]), factors1[i])
                        else:
                            accum1 *= tl.dot(tl.transpose(factors1[i]), factors1[i])
                accum1 = tl.reshape(weights, (-1, 1)) * accum1 * tl.reshape(weights, (1, -1))
                if mask1 is not None:
                    tensor1 = tensor1 * mask1 + tl.cp_to_tensor(
                        (weights, factors1), mask=1 - mask1
                    )

                # Compute accumulation matrix for tensor2
                accum2 = tl.ones((rank, rank), **tl.context(tensor2))
                for i in range(n_modes):
                    if i != mode:
                        if i == non_shared_mode:
                            accum2 *= tl.dot(tl.transpose(factors2[i]), factors2[i])
                        else:
                            accum2 *= tl.dot(tl.transpose(factors2[i]), factors2[i])
                accum2 = tl.reshape(weights, (-1, 1)) * accum2 * tl.reshape(weights, (1, -1))
                if mask2 is not None:
                    tensor2 = tensor2 * mask2 + tl.cp_to_tensor(
                        (weights, factors2), mask=1 - mask2
                    )

                # Compute MTTKRP for both tensors
                mttkrp1 = unfolding_dot_khatri_rao(tensor1, (weights, factors1), mode)
                mttkrp2 = unfolding_dot_khatri_rao(tensor2, (weights, factors2), mode)

                # Combine updates from both tensors
                numerator = tl.clip(mttkrp1 + mttkrp2, a_min=epsilon, a_max=None)
                denominator = tl.clip(
                    tl.dot(factors1[mode], accum1) + tl.dot(factors2[mode], accum2),
                    a_min=epsilon, a_max=None
                )

                # Update shared factor
                new_factor = factors1[mode] * numerator / denominator
                factors1[mode] = new_factor
                factors2[mode] = tl.copy(new_factor)

            if normalize_factors and mode != n_modes - 1:
                weights, factors1 = cp_normalize((weights, factors1))
                weights, factors2 = cp_normalize((weights, factors2))

        # Check convergence
        if tol:
            # Calculate reconstruction errors for both tensors
            unnorml_rec_error1, _, _ = error_calc(
                tensor1, norm_tensor1, weights, factors1,
                sparsity=None, mask=None, mttkrp=mttkrp1 if mode == n_modes - 1 else None
            )
            rec_error1 = unnorml_rec_error1 / norm_tensor1
            rec_errors1.append(rec_error1)

            unnorml_rec_error2, _, _ = error_calc(
                tensor2, norm_tensor2, weights, factors2,
                sparsity=None, mask=None, mttkrp=mttkrp2 if mode == n_modes - 1 else None
            )
            rec_error2 = unnorml_rec_error2 / norm_tensor2
            rec_errors2.append(rec_error2)

            # Use combined error for convergence check
            # Apply balancing weights if requested
            combined_error = (weight1 * rec_error1 + weight2 * rec_error2) / (weight1 + weight2)

            if iteration >= 1:
                prev_combined = (weight1 * rec_errors1[-2] + weight2 * rec_errors2[-2]) / (weight1 + weight2)
                error_decrease = prev_combined - combined_error

                if verbose:
                    print(f"Iteration {iteration}: rec_error1={rec_error1:.6f}, "
                          f"rec_error2={rec_error2:.6f}, combined={combined_error:.6f}, "
                          f"decrease={error_decrease:.6e}")
                    if balance_errors and verbose > 1:
                        print(f"  Balance weights: w1={weight1:.3f}, w2={weight2:.3f}")

                if cvg_criterion == "abs_rec_error":
                    stop_flag = tl.abs(error_decrease) < tol
                elif cvg_criterion == "rec_error":
                    stop_flag = error_decrease < tol
                else:
                    raise ValueError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print(f"Coupled PARAFAC converged after {iteration} iterations")
                    break
            else:
                if verbose:
                    print(f"Initial errors: tensor1={rec_errors1[-1]:.6f}, "
                          f"tensor2={rec_errors2[-1]:.6f}")

    if normalize_factors:
        weights, factors1 = cp_normalize((weights, factors1))
        weights, factors2 = cp_normalize((weights, factors2))

    cp_tensor1 = CPTensor((weights, factors1))
    cp_tensor2 = CPTensor((weights, factors2))

    if return_errors:
        return cp_tensor1, cp_tensor2, (rec_errors1, rec_errors2)
    else:
        return cp_tensor1, cp_tensor2