# -*- coding: utf-8 -*-
# This code is an adaptation built upon the non_negative_parafac function in tensorly.
import numpy as np
import tensorly as tl
from tqdm import tqdm
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
        separate_weights=True,
):
    '''
    Performs coupled non-negative CP decomposition on two tensors.

    Parameters
    ----------
    tensor1 : tensorly.tensor
        First tensor to factorize.

    tensor2 : tensorly.tensor
        Second tensor to factorize.

    rank : int
        Number of components for the factorization.

    non_shared_mode : int
        The mode (dimension) that differs between the two tensors.

    mask1 : tensorly.tensor, default=None
        Mask for the first tensor.

    mask2 : tensorly.tensor, default=None
        Mask for the second tensor.

    n_iter_max : int, default=100
        Maximum number of iterations.

    init : str, default='svd'
        Initialization method. Options are {'svd', 'random'}.

    svd : str, default='truncated_svd'
        SVD function to use.

    tol : float, default=1e-7
        Convergence tolerance.

    random_state : int, default=None
        Random state for reproducibility.

    verbose : bool, default=False
        Whether to print progress.

    normalize_factors : bool, default=True
        Whether to normalize factors.

    return_errors : bool, default=False
        Whether to return reconstruction errors.

    cvg_criterion : str, default='abs_rec_error'
        Convergence criterion. Options are {'abs_rec_error', 'rec_error'}.

    balance_errors : bool, default=True
        Whether to balance errors based on tensor sizes.

    separate_weights : bool, default=True
        Whether to use separate weights for each tensor during optimization.
        If True, each tensor maintains its own weights vector.
        If False, uses shared weights (averaged from both tensors).

    Returns
    -------
    cp_tensor1 : CPTensor
        CP decomposition result for tensor1.

    cp_tensor2 : CPTensor
        CP decomposition result for tensor2.

    errors : tuple, optional
        Reconstruction errors for both tensors, if `return_errors` is True.

    Examples
    --------
    >>> # Two tensors sharing dimensions 0,1,2 but different in dimension 3
    >>> tensor1 = tl.random.random((10, 20, 30, 40))
    >>> tensor2 = tl.random.random((10, 20, 30, 50))
    >>> cp1, cp2 = coupled_non_negative_parafac(tensor1, tensor2, rank=5,
    ...                                        non_shared_mode=3, separate_weights=True)
    '''

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

    if separate_weights:
        # Keep separate weights for each tensor
        if verbose > 1:
            print("Using separate weights for each tensor")
    else:
        # Use shared weights (original behavior)
        if verbose > 1:
            print("Using shared weights (averaged from both tensors)")
        weights_shared = (weights1 + weights2) / 2
        weights1 = weights_shared
        weights2 = tl.copy(weights_shared)

    # Copy shared factors from tensor1 initialization for non-shared modes
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

        # Create update order: non-shared mode first, then shared modes in reverse order
        update_order = []

        # First add the non-shared mode
        update_order.append(non_shared_mode)

        # Then add all shared modes in reverse order (from highest to lowest)
        for mode in reversed(range(n_modes)):
            if mode != non_shared_mode:
                update_order.append(mode)

        if verbose > 1:
            print(f"Update order: {update_order} (non-shared: {non_shared_mode})")

        # Update each mode in the strategic order
        for mode in update_order:
            if verbose > 1:
                print(f"Mode {mode} of {n_modes}")

            if mode == non_shared_mode:
                # Update non-shared mode separately for each tensor

                # Update for tensor1
                accum1 = tl.ones((rank, rank), **tl.context(tensor1))
                for i in range(n_modes):
                    if i != mode:
                        accum1 *= tl.dot(tl.transpose(factors1[i]), factors1[i])
                accum1 = tl.reshape(weights1, (-1, 1)) * accum1 * tl.reshape(weights1, (1, -1))
                if mask1 is not None:
                    tensor1 = tensor1 * mask1 + tl.cp_to_tensor(
                        (weights1, factors1), mask=1 - mask1
                    )

                mttkrp1 = unfolding_dot_khatri_rao(tensor1, (weights1, factors1), mode)
                numerator1 = tl.clip(mttkrp1, a_min=epsilon, a_max=None)
                denominator1 = tl.clip(tl.dot(factors1[mode], accum1), a_min=epsilon, a_max=None)
                factors1[mode] = factors1[mode] * numerator1 / denominator1

                # Update for tensor2
                accum2 = tl.ones((rank, rank), **tl.context(tensor2))
                for i in range(n_modes):
                    if i != mode:
                        accum2 *= tl.dot(tl.transpose(factors2[i]), factors2[i])
                accum2 = tl.reshape(weights2, (-1, 1)) * accum2 * tl.reshape(weights2, (1, -1))
                if mask2 is not None:
                    tensor2 = tensor2 * mask2 + tl.cp_to_tensor(
                        (weights2, factors2), mask=1 - mask2
                    )

                mttkrp2 = unfolding_dot_khatri_rao(tensor2, (weights2, factors2), mode)
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
                accum1 = tl.reshape(weights1, (-1, 1)) * accum1 * tl.reshape(weights1, (1, -1))
                if mask1 is not None:
                    tensor1 = tensor1 * mask1 + tl.cp_to_tensor(
                        (weights1, factors1), mask=1 - mask1
                    )

                # Compute accumulation matrix for tensor2
                accum2 = tl.ones((rank, rank), **tl.context(tensor2))
                for i in range(n_modes):
                    if i != mode:
                        if i == non_shared_mode:
                            accum2 *= tl.dot(tl.transpose(factors2[i]), factors2[i])
                        else:
                            accum2 *= tl.dot(tl.transpose(factors2[i]), factors2[i])
                accum2 = tl.reshape(weights2, (-1, 1)) * accum2 * tl.reshape(weights2, (1, -1))
                if mask2 is not None:
                    tensor2 = tensor2 * mask2 + tl.cp_to_tensor(
                        (weights2, factors2), mask=1 - mask2
                    )

                # Compute MTTKRP for both tensors
                mttkrp1 = unfolding_dot_khatri_rao(tensor1, (weights1, factors1), mode)
                mttkrp2 = unfolding_dot_khatri_rao(tensor2, (weights2, factors2), mode)

                # Combine updates from both tensors
                numerator = tl.clip((mttkrp1 + mttkrp2)/2., a_min=epsilon, a_max=None)
                denominator = tl.clip(
                    (tl.dot(factors1[mode], accum1) + tl.dot(factors2[mode], accum2))/2.,
                    a_min=epsilon, a_max=None
                )

                # Update shared factor
                new_factor = factors1[mode] * numerator / denominator
                factors1[mode] = new_factor
                factors2[mode] = tl.copy(new_factor)

            # Normalize factors and weights separately for each tensor if requested
            if normalize_factors and mode != n_modes - 1:
                if separate_weights:
                    weights1, factors1 = cp_normalize((weights1, factors1))
                    weights2, factors2 = cp_normalize((weights2, factors2))
                else:
                    # For shared weights, normalize both and then average weights again
                    weights1, factors1 = cp_normalize((weights1, factors1))
                    weights2, factors2 = cp_normalize((weights2, factors2))
                    weights_shared = (weights1 + weights2) / 2
                    weights1 = weights_shared
                    weights2 = tl.copy(weights_shared)

        # Check convergence
        if tol:
            # Calculate reconstruction errors for both tensors
            unnorml_rec_error1, _, _ = error_calc(
                tensor1, norm_tensor1, weights1, factors1,
                sparsity=None, mask=None, mttkrp=mttkrp1 if mode == n_modes - 1 else None
            )
            rec_error1 = unnorml_rec_error1 / norm_tensor1
            rec_errors1.append(rec_error1)

            unnorml_rec_error2, _, _ = error_calc(
                tensor2, norm_tensor2, weights2, factors2,
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
                    if separate_weights:
                        print(f"Iteration {iteration}: rec_error1={rec_error1:.6f}, "
                              f"rec_error2={rec_error2:.6f}, combined={combined_error:.6f}, "
                              f"decrease={error_decrease:.6e} [separate weights]")
                    else:
                        print(f"Iteration {iteration}: rec_error1={rec_error1:.6f}, "
                              f"rec_error2={rec_error2:.6f}, combined={combined_error:.6f}, "
                              f"decrease={error_decrease:.6e} [shared weights]")
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

    # Final normalization
    if normalize_factors:
        if separate_weights:
            weights1, factors1 = cp_normalize((weights1, factors1))
            weights2, factors2 = cp_normalize((weights2, factors2))
        else:
            weights1, factors1 = cp_normalize((weights1, factors1))
            weights2, factors2 = cp_normalize((weights2, factors2))
            weights_shared = (weights1 + weights2) / 2
            weights1 = weights_shared
            weights2 = tl.copy(weights_shared)

    cp_tensor1 = CPTensor((weights1, factors1))
    cp_tensor2 = CPTensor((weights2, factors2))

    if return_errors:
        return cp_tensor1, cp_tensor2, (rec_errors1, rec_errors2)
    else:
        return cp_tensor1, cp_tensor2


def _compute_coupled_tensor_factorization(tensor1, tensor2, rank, non_shared_mode, mask1=None, mask2=None,
                                          tf_type='coupled_non_negative_cp', init='svd', svd='truncated_svd',
                                          random_state=None, n_iter_max=100, tol=10e-7, verbose=False,
                                          balance_errors=True, **kwargs):
    '''Performs the Coupled Tensor Factorization

    Parameters
    ----------
    tensor1 : ndarray list
        First tensor to factorize.

    tensor2 : ndarray list
        Second tensor to factorize.

    rank : int
        Rank of the Tensor Factorization (number of factors).

    non_shared_mode : int
        The mode (dimension) that differs between the two tensors.

    mask1 : ndarray list, default=None
        Mask for the first tensor.

    mask2 : ndarray list, default=None
        Mask for the second tensor.

    tf_type : str, default='coupled_non_negative_cp'
        Type of Tensor Factorization. Currently only supports 'coupled_non_negative_cp'.

    init : str, default='svd'
        Initialization method. Options are {'svd', 'random'}.

    svd : str, default='truncated_svd'
        Function to use to compute the SVD.

    random_state : int, default=None
        Seed for randomization.

    n_iter_max : int, default=100
        Maximum number of iterations.

    tol : float, default=10e-7
        Convergence tolerance.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    balance_errors : bool, default=True
        Whether to balance errors based on tensor sizes.

    **kwargs : dict
        Extra arguments for the tensor factorization.

    Returns
    -------
    cp_tf1 : CPTensor
        CP decomposition result for tensor1.

    cp_tf2 : CPTensor
        CP decomposition result for tensor2.

    errors : tuple, optional
        Reconstruction errors for both tensors, if `return_errors` is True.
    '''
    # For returning errors
    return_errors = False
    if kwargs is not None:
        if 'return_errors' not in kwargs.keys():
            kwargs['return_errors'] = return_errors
    else:
        kwargs = {'return_errors': return_errors}


    if tf_type == 'coupled_non_negative_cp':
        result = coupled_non_negative_parafac(
            tensor1=tensor1,
            tensor2=tensor2,
            rank=rank,
            non_shared_mode=non_shared_mode,
            mask1=mask1,
            mask2=mask2,
            init='random' if (mask1 is not None or mask2 is not None) else init,
            svd=svd,
            random_state=random_state,
            n_iter_max=n_iter_max,
            tol=tol,
            verbose=verbose,
            balance_errors=balance_errors,
            **kwargs
        )
    else:
        raise ValueError('Not a valid tf_type for coupled factorization.')

    return result


def _run_coupled_elbow_analysis(tensor1, tensor2, non_shared_mode, upper_rank=50, tf_type='coupled_non_negative_cp',
                                init='svd', svd='truncated_svd', random_state=None, mask1=None, mask2=None,
                                n_iter_max=100, tol=10e-7, verbose=False, balance_errors=True,
                                disable_pbar=False, **kwargs):
    '''Performs a coupled elbow analysis with just one run of a tensor factorization for each rank

    Parameters
    ----------
    tensor1 : ndarray list
        First tensor to factorize.

    tensor2 : ndarray list
        Second tensor to factorize.

    non_shared_mode : int
        The mode (dimension) that differs between the two tensors.

    upper_rank : int, default=50
        Upper bound of ranks to explore with the elbow analysis.

    tf_type : str, default='coupled_non_negative_cp'
        Type of Tensor Factorization.

    init : str, default='svd'
        Initialization method. {'svd', 'random'}

    svd : str, default='truncated_svd'
        Function to use to compute the SVD.

    random_state : int, default=None
        Seed for randomization.

    mask1 : ndarray list, default=None
        Mask for the first tensor.

    mask2 : ndarray list, default=None
        Mask for the second tensor.

    n_iter_max : int, default=100
        Maximum number of iterations.

    tol : float, default=10e-7
        Convergence tolerance.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    balance_errors : bool, default=True
        Whether to balance errors based on tensor sizes.

    disable_pbar : boolean, default=False
        Whether displaying a tqdm progress bar or not.

    **kwargs : dict
        Extra arguments for the tensor factorization.

    Returns
    -------
    loss : list
        List of tuples with (rank, combined_error) coordinates for the elbow analysis.
    '''
    if kwargs is None:
        kwargs = {'return_errors': True}
    else:
        kwargs['return_errors'] = True

    loss = []
    for r in tqdm(range(1, upper_rank + 1), disable=disable_pbar):
        cp1, cp2, (errors1, errors2) = _compute_coupled_tensor_factorization(
            tensor1=tensor1,
            tensor2=tensor2,
            rank=r,
            non_shared_mode=non_shared_mode,
            mask1=mask1,
            mask2=mask2,
            tf_type=tf_type,
            init=init,
            svd=svd,
            random_state=random_state,
            n_iter_max=n_iter_max,
            tol=tol,
            verbose=verbose,
            balance_errors=balance_errors,
            **kwargs
        )

        # Calculate combined error using _compute_norm_error when masks are present
        if mask1 is None:
            error1 = tl.to_numpy(errors1[-1])
        else:
            from cell2cell.tensor.factorization import _compute_norm_error
            error1 = _compute_norm_error(tensor1, cp1, mask1)

        if mask2 is None:
            error2 = tl.to_numpy(errors2[-1])
        else:
            from cell2cell.tensor.factorization import _compute_norm_error
            error2 = _compute_norm_error(tensor2, cp2, mask2)

        # Calculate combined error with balancing
        N1 = tensor1.shape[non_shared_mode]
        N2 = tensor2.shape[non_shared_mode]

        if balance_errors:
            total_N = N1 + N2
            weight1 = total_N / N1
            weight2 = total_N / N2
            combined_error = (weight1 * error1 + weight2 * error2) / (weight1 + weight2)
        else:
            combined_error = (error1 + error2) / 2

        loss.append((r, combined_error))

    return loss


def _multiple_runs_coupled_elbow_analysis(tensor1, tensor2, non_shared_mode, upper_rank=50, runs=10,
                                          tf_type='coupled_non_negative_cp', init='svd', svd='truncated_svd',
                                          metric='error', random_state=None, mask1=None, mask2=None,
                                          n_iter_max=100, tol=10e-7, verbose=False, balance_errors=True, **kwargs):
    '''Performs a coupled elbow analysis with multiple runs of tensor factorization for each rank

    Parameters
    ----------
    tensor1 : ndarray list
        First tensor to factorize.

    tensor2 : ndarray list
        Second tensor to factorize.

    non_shared_mode : int
        The mode (dimension) that differs between the two tensors.

    upper_rank : int, default=50
        Upper bound of ranks to explore with the elbow analysis.

    runs : int, default=10
        Number of tensor factorization performed for a given rank.

    tf_type : str, default='coupled_non_negative_cp'
        Type of Tensor Factorization.

    init : str, default='svd'
        Initialization method. {'svd', 'random'}

    svd : str, default='truncated_svd'
        Function to use to compute the SVD.

    metric : str, default='error'
        Metric to perform the elbow analysis. Currently only supports 'error'.

    random_state : int, default=None
        Seed for randomization.

    mask1 : ndarray list, default=None
        Mask for the first tensor.

    mask2 : ndarray list, default=None
        Mask for the second tensor.

    n_iter_max : int, default=100
        Maximum number of iterations.

    tol : float, default=10e-7
        Convergence tolerance.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    balance_errors : bool, default=True
        Whether to balance errors based on tensor sizes.

    **kwargs : dict
        Extra arguments for the tensor factorization.

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

    if metric == 'similarity':
        raise NotImplementedError("Similarity metric for coupled tensors not yet implemented")

    all_loss = []
    for r in tqdm(range(1, upper_rank + 1)):
        run_errors = []
        for run in range(runs):
            if random_state is not None:
                rs = random_state + run
            else:
                rs = None

            cp1, cp2, (errors1, errors2) = _compute_coupled_tensor_factorization(
                tensor1=tensor1,
                tensor2=tensor2,
                rank=r,
                non_shared_mode=non_shared_mode,
                mask1=mask1,
                mask2=mask2,
                tf_type=tf_type,
                init=init,
                svd=svd,
                random_state=rs,
                n_iter_max=n_iter_max,
                tol=tol,
                verbose=verbose,
                balance_errors=balance_errors,
                **kwargs
            )

            # Calculate combined error using _compute_norm_error when masks are present
            if mask1 is None:
                error1 = tl.to_numpy(errors1[-1])
            else:
                from cell2cell.tensor.factorization import _compute_norm_error
                error1 = _compute_norm_error(tensor1, cp1, mask1)

            if mask2 is None:
                error2 = tl.to_numpy(errors2[-1])
            else:
                from cell2cell.tensor.factorization import _compute_norm_error
                error2 = _compute_norm_error(tensor2, cp2, mask2)

            # Calculate combined error with balancing
            N1 = tensor1.shape[non_shared_mode]
            N2 = tensor2.shape[non_shared_mode]

            if balance_errors:
                total_N = N1 + N2
                weight1 = total_N / N1
                weight2 = total_N / N2
                combined_error = (weight1 * error1 + weight2 * error2) / (weight1 + weight2)
            else:
                combined_error = (error1 + error2) / 2

            run_errors.append(combined_error)

        all_loss.append(run_errors)

    all_loss = np.array(all_loss).T
    return all_loss