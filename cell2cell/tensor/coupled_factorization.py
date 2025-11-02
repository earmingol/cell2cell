# -*- coding: utf-8 -*-
# This code is an adaptation built upon the non_negative_parafac function in tensorly.
import numpy as np
import scipy.spatial as sp
import tensorly as tl
from tqdm import tqdm
from tensorly.decomposition._cp import initialize_cp, error_calc
from tensorly.cp_tensor import (
    CPTensor,
    unfolding_dot_khatri_rao,
    cp_normalize,
    validate_cp_rank,
)

from cell2cell.tensor.metrics import pairwise_correlation_index


def _process_mode_mapping(tensor1, tensor2, mode_mapping):
    '''
    Processes mode mapping input, handling backward compatibility

    Parameters
    ----------
    tensor1, tensor2 : tensor objects or raw tensors
        Can be either tensorly tensors or objects with .tensor attribute

    mode_mapping : dict or int/list
        Mode mapping specification. Can be:
        - dict: {'shared': [(t1_mode, t2_mode), ...]}     # Pairs of shared modes
        - int/list: non_shared_modes (for backward compatibility with same-dimension tensors)

    Returns
    -------
    mode_mapping : dict
        Processed mode mapping in dict format
    '''

    if isinstance(mode_mapping, dict):
        return mode_mapping

    # Handle backward compatibility with non_shared_modes
    ndim1 = tl.ndim(tensor1)
    ndim2 = tl.ndim(tensor2)

    if ndim1 != ndim2:
        raise ValueError(
            "When tensors have different dimensions, mode_mapping must be a dict. "
            "For backward compatibility with non_shared_modes, both tensors must have same dimensions."
        )

    # Convert non_shared_modes format to mode_mapping format
    if isinstance(mode_mapping, int):
        non_shared_modes = [mode_mapping]
    elif isinstance(mode_mapping, (list, tuple)):
        non_shared_modes = list(mode_mapping)
    else:
        raise ValueError("mode_mapping must be dict, int, list, or tuple")

    # Remove duplicates with warning
    if len(set(non_shared_modes)) != len(non_shared_modes):
        original = non_shared_modes.copy()
        non_shared_modes = list(set(non_shared_modes))
        print(
            f"Warning: Duplicate modes found in non_shared_modes {original}. Using deduplicated version: {non_shared_modes}")

    # Create mode_mapping dict - only specify shared pairs
    shared_pairs = [(i, i) for i in range(ndim1) if i not in non_shared_modes]
    return {'shared': shared_pairs}


def _validate_tensors(tensor1, tensor2, mode_mapping):
    '''
    Validates that the two tensors are compatible for coupled factorization

    Parameters
    ----------
    tensor1, tensor2 : tensor objects or raw tensors
        Can be either tensorly tensors or objects with .tensor attribute

    mode_mapping : dict
        Mode mapping specification in dict format: {'shared': [(t1_mode, t2_mode), ...]}

    Raises
    ------
    ValueError
        If tensors are incompatible based on mode mapping

    '''
    # Handle both raw tensors and tensor objects
    t1 = tensor1.tensor if hasattr(tensor1, 'tensor') else tensor1
    t2 = tensor2.tensor if hasattr(tensor2, 'tensor') else tensor2

    ndim1 = tl.ndim(t1)
    ndim2 = tl.ndim(t2)

    # Parse mode mapping
    shared_pairs = mode_mapping.get('shared', [])

    # Automatically derive tensor-specific modes
    shared_t1_modes = set([pair[0] for pair in shared_pairs])
    shared_t2_modes = set([pair[1] for pair in shared_pairs])

    # Validate mode coverage
    if not shared_t1_modes.issubset(set(range(ndim1))):
        raise ValueError(f"Shared modes contain invalid tensor1 modes: {sorted(shared_t1_modes - set(range(ndim1)))}")
    if not shared_t2_modes.issubset(set(range(ndim2))):
        raise ValueError(f"Shared modes contain invalid tensor2 modes: {sorted(shared_t2_modes - set(range(ndim2)))}")

    if len(shared_pairs) == 0:
        raise ValueError("At least one mode must be shared between tensors")

    # Check shared dimensions match in size
    for t1_mode, t2_mode in shared_pairs:
        if t1.shape[t1_mode] != t2.shape[t2_mode]:
            raise ValueError(f"Shared modes must have same size: tensor1 mode {t1_mode} "
                             f"({t1.shape[t1_mode]}) vs tensor2 mode {t2_mode} ({t2.shape[t2_mode]})")

    # Check element names match for shared dimensions (only for tensor objects)
    if hasattr(tensor1, 'order_names') and hasattr(tensor2, 'order_names'):
        for t1_mode, t2_mode in shared_pairs:
            if tensor1.order_names[t1_mode] != tensor2.order_names[t2_mode]:
                raise ValueError(
                    f"Element names must match for shared dimensions: tensor1 mode {t1_mode} vs tensor2 mode {t2_mode}")


def _compute_balancing_weights(tensor1, tensor2, mode_mapping, balance_errors=True, manual_weights=(0.5, 0.5)):
    '''
    Compute or retrieve balancing weights for coupled tensor factorization.

    Parameters
    ----------
    tensor1 : tensorly.tensor
        First tensor

    tensor2 : tensorly.tensor
        Second tensor

    mode_mapping : dict
        Mode mapping specification

    balance_errors : bool, default=True
        Whether to automatically balance errors based on tensor sizes. If not,
        automatic weight1 = weight2 = 0.5 is used.

    manual_weights : tuple, default=(0.5, 0.5)
        Manual weights (weight1, weight2) for importance of tensors in the factorization.
        Weights should be positive. Example: (2.0, 1.0) gives tensor1 twice
        the importance of tensor2 in both the factorization and the combined error metric.
        If None, automatic weight calculation is performed to have weight1 and weight2
        inversely proportional to non-shared mode dimensions of each tensor.

    Returns
    -------
    weight1, weight2 : float
        Weights for balancing errors
    '''
    if balance_errors:
        # If manual weights provided, use them directly
        if manual_weights is not None:
            if not isinstance(manual_weights, (tuple, list)) or len(manual_weights) != 2:
                raise ValueError("manual_weights must be a tuple/list of 2 positive numbers (weight1, weight2)")
            weight1, weight2 = manual_weights
            if weight1 <= 0 or weight2 <= 0:
                raise ValueError("manual_weights must be positive")
            return float(weight1), float(weight2)

        # If manual not provided, use weights inversely proportional to dimensions of non-shared modes
        shared_pairs = mode_mapping.get('shared', [])
        shared_t1_modes = set([pair[0] for pair in shared_pairs])
        shared_t2_modes = set([pair[1] for pair in shared_pairs])

        ndim1 = tl.ndim(tensor1)
        ndim2 = tl.ndim(tensor2)
        tensor1_only = [i for i in range(ndim1) if i not in shared_t1_modes]
        tensor2_only = [i for i in range(ndim2) if i not in shared_t2_modes]

        nonshared_size1 = np.prod([tensor1.shape[i] for i in tensor1_only]) if tensor1_only else 1
        nonshared_size2 = np.prod([tensor2.shape[i] for i in tensor2_only]) if tensor2_only else 1
        total_nonshared = nonshared_size1 + nonshared_size2

        if total_nonshared > 0:
            weight1 = total_nonshared / nonshared_size1 if nonshared_size1 > 0 else 0.5
            weight2 = total_nonshared / nonshared_size2 if nonshared_size2 > 0 else 0.5
        else:
            weight1 = weight2 = 0.5
    else:
        weight1 = weight2 = 0.5
    return weight1, weight2


def coupled_non_negative_parafac(
        tensor1,
        tensor2,
        rank,
        mode_mapping,
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
        manual_weights=(0.5, 0.5),
        separate_weights=True,
):
    '''
    Performs coupled non-negative CP decomposition on two tensors with flexible mode mapping.

    Parameters
    ----------
    tensor1 : tensorly.tensor
        First tensor to factorize.

    tensor2 : tensorly.tensor
        Second tensor to factorize.

    rank : int
        Number of components for the factorization.

    mode_mapping : dict or int/list (for backward compatibility)
        Mode mapping specification. Can be:
        - dict: {'shared': [(t1_mode, t2_mode), ...]}     # Pairs of shared modes
        - int/list: non_shared_modes (for backward compatibility with same-dimension tensors)

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

    manual_weights : tuple, default=(0.5, 0.5)
        Manual weights (weight1, weight2) for importance of tensors in the factorization.
        Weights should be positive. Example: (2.0, 1.0) gives tensor1 twice
        the importance of tensor2 in both the factorization and the combined error metric.
        If None, automatic weight calculation is performed to have weight1 and weight2
        inversely proportional to non-shared mode dimensions of each tensor.

    separate_weights : bool, default=True
        Whether to use separate weights for each tensor during optimization.

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
    >>> # Two tensors with different dimensions but shared modes 0,1
    >>> tensor1 = tl.random.random((10, 20, 30, 40))
    >>> tensor2 = tl.random.random((10, 20, 50))
    >>> mode_mapping = {'shared': [(0, 0), (1, 1)]}
    >>> cp1, cp2 = coupled_non_negative_parafac(tensor1, tensor2, rank=5,
    ...                                         mode_mapping=mode_mapping)
    >>>
    >>> # Using manual weights to prioritize tensor1
    >>> cp1, cp2 = coupled_non_negative_parafac(tensor1, tensor2, rank=5,
    ...                                         mode_mapping=mode_mapping,
    ...                                         manual_weights=(2.0, 1.0))
    '''

    epsilon = tl.eps(tensor1.dtype)

    # Validate inputs
    ndim1 = tl.ndim(tensor1)
    ndim2 = tl.ndim(tensor2)

    # Parse and validate mode mapping
    shared_pairs = mode_mapping.get('shared', [])

    # Automatically derive tensor-specific modes
    shared_t1_modes = set([pair[0] for pair in shared_pairs])
    shared_t2_modes = set([pair[1] for pair in shared_pairs])

    tensor1_only = [i for i in range(ndim1) if i not in shared_t1_modes]
    tensor2_only = [i for i in range(ndim2) if i not in shared_t2_modes]

    # Validate mode mapping completeness
    all_t1_modes = shared_t1_modes | set(tensor1_only)
    all_t2_modes = shared_t2_modes | set(tensor2_only)

    if all_t1_modes != set(range(ndim1)):
        raise ValueError(f"Shared modes must be valid tensor1 modes, got tensor1 modes {sorted(shared_t1_modes)}")
    if all_t2_modes != set(range(ndim2)):
        raise ValueError(f"Shared modes must be valid tensor2 modes, got tensor2 modes {sorted(shared_t2_modes)}")

    if len(shared_pairs) == 0:
        raise ValueError("At least one mode must be shared between tensors")

    # Check shared dimensions match
    for t1_mode, t2_mode in shared_pairs:
        if tensor1.shape[t1_mode] != tensor2.shape[t2_mode]:
            raise ValueError(f"Shared modes must have same size: tensor1 mode {t1_mode} "
                             f"({tensor1.shape[t1_mode]}) vs tensor2 mode {t2_mode} ({tensor2.shape[t2_mode]})")

    # Validate rank
    rank = validate_cp_rank(tl.shape(tensor1), rank=rank)

    # Calculate balancing weights (either manual or automatic)
    weight1, weight2 = _compute_balancing_weights(
        tensor1, tensor2, mode_mapping, balance_errors, manual_weights
    )

    if verbose > 1 and balance_errors:
        print(f"Using weights: w1={weight1:.3f}, w2={weight2:.3f}")

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
        if verbose > 1:
            print("Using separate weights for each tensor")
    else:
        if verbose > 1:
            print("Using shared weights (averaged from both tensors)")
        weights_shared = (weights1 + weights2) / 2
        weights1 = weights_shared
        weights2 = tl.copy(weights_shared)

    # Initialize shared factors from tensor1
    for t1_mode, t2_mode in shared_pairs:
        factors2[t2_mode] = tl.copy(factors1[t1_mode])

    # Store norms for convergence checking
    norm_tensor1 = tl.norm(tensor1, 2)
    norm_tensor2 = tl.norm(tensor2, 2)

    rec_errors1 = []
    rec_errors2 = []

    for iteration in range(n_iter_max):
        if verbose > 1:
            print(f"Starting iteration {iteration + 1}")

        # Create strategic update order: tensor-specific modes first, then shared modes
        update_order = []

        # First update tensor-specific modes
        for mode in tensor1_only:
            update_order.append(('tensor1_only', mode, None))
        for mode in tensor2_only:
            update_order.append(('tensor2_only', None, mode))

        # Then update shared modes (in reverse order for stability)
        for t1_mode, t2_mode in reversed(shared_pairs):
            update_order.append(('shared', t1_mode, t2_mode))

        if verbose > 1:
            print(f"Update order: {[(item[0], item[1] if item[1] is not None else item[2]) for item in update_order]}")

        # Update each mode according to the strategy
        for mode_type, t1_mode, t2_mode in update_order:
            if verbose > 1:
                if mode_type == 'shared':
                    print(f"Shared modes: tensor1[{t1_mode}] <-> tensor2[{t2_mode}]")
                elif mode_type == 'tensor1_only':
                    print(f"Tensor1-only mode: {t1_mode}")
                else:
                    print(f"Tensor2-only mode: {t2_mode}")

            if mode_type == 'tensor1_only':
                # Update tensor1-specific mode
                mode = t1_mode
                accum1 = tl.ones((rank, rank), **tl.context(tensor1))
                for i in range(ndim1):
                    if i != mode:
                        accum1 *= tl.dot(tl.transpose(factors1[i]), factors1[i])
                accum1 = tl.reshape(weights1, (-1, 1)) * accum1 * tl.reshape(weights1, (1, -1))

                if mask1 is not None:
                    tensor1 = tensor1 * mask1 + tl.cp_to_tensor((weights1, factors1), mask=1 - mask1)

                mttkrp1 = unfolding_dot_khatri_rao(tensor1, (weights1, factors1), mode)
                numerator1 = tl.clip(mttkrp1, a_min=epsilon, a_max=None)
                denominator1 = tl.clip(tl.dot(factors1[mode], accum1), a_min=epsilon, a_max=None)
                factors1[mode] = factors1[mode] * numerator1 / denominator1

            elif mode_type == 'tensor2_only':
                # Update tensor2-specific mode
                mode = t2_mode
                accum2 = tl.ones((rank, rank), **tl.context(tensor2))
                for i in range(ndim2):
                    if i != mode:
                        accum2 *= tl.dot(tl.transpose(factors2[i]), factors2[i])
                accum2 = tl.reshape(weights2, (-1, 1)) * accum2 * tl.reshape(weights2, (1, -1))

                if mask2 is not None:
                    tensor2 = tensor2 * mask2 + tl.cp_to_tensor((weights2, factors2), mask=1 - mask2)

                mttkrp2 = unfolding_dot_khatri_rao(tensor2, (weights2, factors2), mode)
                numerator2 = tl.clip(mttkrp2, a_min=epsilon, a_max=None)
                denominator2 = tl.clip(tl.dot(factors2[mode], accum2), a_min=epsilon, a_max=None)
                factors2[mode] = factors2[mode] * numerator2 / denominator2

            else:  # shared mode
                # Update shared modes using combined information from both tensors
                # Compute accumulation for tensor1
                accum1 = tl.ones((rank, rank), **tl.context(tensor1))
                for i in range(ndim1):
                    if i != t1_mode:
                        accum1 *= tl.dot(tl.transpose(factors1[i]), factors1[i])
                accum1 = tl.reshape(weights1, (-1, 1)) * accum1 * tl.reshape(weights1, (1, -1))

                if mask1 is not None:
                    tensor1 = tensor1 * mask1 + tl.cp_to_tensor((weights1, factors1), mask=1 - mask1)

                # Compute accumulation for tensor2
                accum2 = tl.ones((rank, rank), **tl.context(tensor2))
                for i in range(ndim2):
                    if i != t2_mode:
                        accum2 *= tl.dot(tl.transpose(factors2[i]), factors2[i])
                accum2 = tl.reshape(weights2, (-1, 1)) * accum2 * tl.reshape(weights2, (1, -1))

                if mask2 is not None:
                    tensor2 = tensor2 * mask2 + tl.cp_to_tensor((weights2, factors2), mask=1 - mask2)

                # Compute MTTKRP for both tensors
                mttkrp1 = unfolding_dot_khatri_rao(tensor1, (weights1, factors1), t1_mode)
                mttkrp2 = unfolding_dot_khatri_rao(tensor2, (weights2, factors2), t2_mode)

                # Combine updates from both tensors
                numerator = tl.clip(
                    (weight1 * mttkrp1 + weight2 * mttkrp2) / (weight1 + weight2),
                    a_min=epsilon, a_max=None
                )
                denominator = tl.clip(
                    (weight1 * tl.dot(factors1[t1_mode], accum1) +
                     weight2 * tl.dot(factors2[t2_mode], accum2)) / (weight1 + weight2),
                    a_min=epsilon, a_max=None
                )

                # Update shared factor
                new_factor = factors1[t1_mode] * numerator / denominator
                factors1[t1_mode] = new_factor
                factors2[t2_mode] = tl.copy(new_factor)

        # Normalize factors if requested
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

        # Check convergence
        if tol:
            unnorml_rec_error1, _, _ = error_calc(
                tensor1, norm_tensor1, weights1, factors1,
                sparsity=None, mask=None
            )
            rec_error1 = unnorml_rec_error1 / norm_tensor1
            rec_errors1.append(rec_error1)

            unnorml_rec_error2, _, _ = error_calc(
                tensor2, norm_tensor2, weights2, factors2,
                sparsity=None, mask=None
            )
            rec_error2 = unnorml_rec_error2 / norm_tensor2
            rec_errors2.append(rec_error2)

            # Use combined error for convergence check
            combined_error = (weight1 * rec_error1 + weight2 * rec_error2) / (weight1 + weight2)

            if iteration >= 1:
                prev_combined = (weight1 * rec_errors1[-2] + weight2 * rec_errors2[-2]) / (weight1 + weight2)
                error_decrease = prev_combined - combined_error

                if verbose:
                    print(f"Iteration {iteration}: rec_error1={rec_error1:.6f}, "
                          f"rec_error2={rec_error2:.6f}, combined={combined_error:.6f}, "
                          f"decrease={error_decrease:.6e}")
                    if (balance_errors or manual_weights is not None) and verbose > 1:
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


def _compute_coupled_tensor_factorization(tensor1, tensor2, rank, mode_mapping, mask1=None, mask2=None,
                                          tf_type='coupled_non_negative_cp', init='svd', svd='truncated_svd',
                                          random_state=None, n_iter_max=100, tol=10e-7, verbose=False,
                                          balance_errors=True, manual_weights=(0.5, 0.5), **kwargs):
    '''Performs the Coupled Tensor Factorization with flexible mode mapping'''

    if kwargs is None:
        kwargs = {'return_errors': False}
    if 'return_errors' not in kwargs.keys():
        kwargs['return_errors'] = False

    if tf_type == 'coupled_non_negative_cp':
        result = coupled_non_negative_parafac(
            tensor1=tensor1,
            tensor2=tensor2,
            rank=rank,
            mode_mapping=mode_mapping,
            mask1=mask1,
            mask2=mask2,
            init='random' if (mask1 is not None or mask2 is not None) else init,
            svd=svd,
            random_state=random_state,
            n_iter_max=n_iter_max,
            tol=tol,
            verbose=verbose,
            balance_errors=balance_errors,
            manual_weights=manual_weights,
            **kwargs
        )
    else:
        raise ValueError('Not a valid tf_type for coupled factorization.')

    return result


def _run_coupled_elbow_analysis(tensor1, tensor2, mode_mapping, upper_rank=50, tf_type='coupled_non_negative_cp',
                                init='svd', svd='truncated_svd', random_state=None, mask1=None, mask2=None,
                                n_iter_max=100, tol=10e-7, verbose=False, balance_errors=True,
                                manual_weights=(0.5, 0.5), disable_pbar=False, **kwargs):
    '''
    Performs a coupled elbow analysis with mode mapping

    Parameters
    ----------
    tensor1 : tensorly.tensor
        First tensor to factorize.

    tensor2 : tensorly.tensor
        Second tensor to factorize.

    mode_mapping : dict or int/list (for backward compatibility)
        Mode mapping specification. Can be:
        - dict: {'shared': [(t1_mode, t2_mode), ...]}     # Pairs of shared modes
        - int/list: non_shared_modes (for backward compatibility with same-dimension tensors)

    upper_rank : int, default=50
        Maximum rank to evaluate.

    init : str, default='svd'
        Initialization method for computing the Tensor Factorization.
        {'svd', 'random'}

    svd : str, default='truncated_svd'
        Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS

    random_state : int, default=None
        Seed for randomization.

    mask1 : tensorly.tensor, default=None
        Mask for the first tensor. Helps avoiding missing values during a
        tensor factorization. A mask should be a boolean array of the same
        shape as the original tensor and should be 0 where the values are missing and 1 everywhere else.

    mask2 : tensorly.tensor, default=None
        Mask for the second tensor. Helps avoiding missing values during a
        tensor factorization. A mask should be a boolean array of the same
        shape as the original tensor and should be 0 where the values are missing and 1 everywhere else.

    n_iter_max : int, default=100
        Maximum number of iteration to reach an optimal solution with the
        decomposition algorithm. Higher `n_iter_max`helps to improve the solution
        obtained from the decomposition, but it takes longer to run.

    tol : float, default=10e-7
        Tolerance for the decomposition algorithm to stop when the variation in
        the reconstruction error is less than the tolerance. Lower `tol` helps
        to improve the solution obtained from the decomposition, but it takes
        longer to run.

    balance_errors : boolean, default=True
        Whether to balance the errors from each tensor based on their sizes
        during the elbow analysis. This helps to avoid bias towards larger tensors.

    manual_weights : tuple, default=(0.5, 0.5)
        Manual weights (weight1, weight2) for importance of tensors in the factorization.
        Weights should be positive. Example: (2.0, 1.0) gives tensor1 twice
        the importance of tensor2 in both the factorization and the combined error metric.
        If None, automatic weight calculation is performed to have weight1 and weight2
        inversely proportional to non-shared mode dimensions of each tensor.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    disable_pbar : boolean, default=False
        Whether displaying a tqdm progress bar or not.

    **kwargs : dict
        Extra arguments for the tensor factorization according to inputs in tensorly.

    Returns
    -------
    loss : dict
        Dictionary with keys 'tensor1', 'tensor2', and 'combined', each containing
        a list of (rank, error) tuples for the respective errors.
    '''

    if kwargs is None:
        kwargs = {'return_errors': True}
    else:
        kwargs['return_errors'] = True

    loss_t1 = []
    loss_t2 = []
    loss_combined = []

    for r in tqdm(range(1, upper_rank + 1), disable=disable_pbar):
        cp1, cp2, (errors1, errors2) = _compute_coupled_tensor_factorization(
            tensor1=tensor1,
            tensor2=tensor2,
            rank=r,
            mode_mapping=mode_mapping,
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
            manual_weights=manual_weights,
            **kwargs
        )

        # Calculate individual errors
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
        weight1, weight2 = _compute_balancing_weights(tensor1, tensor2, mode_mapping, balance_errors, manual_weights)
        combined_error = (weight1 * error1 + weight2 * error2) / (weight1 + weight2)

        loss_t1.append((r, error1))
        loss_t2.append((r, error2))
        loss_combined.append((r, combined_error))

    return {
        'tensor1': loss_t1,
        'tensor2': loss_t2,
        'combined': loss_combined
    }


def _create_combined_factors_dict(factors1_dict, factors2_dict, mode_mapping):
    '''
    Creates a combined factors dictionary from two separate factor dictionaries
    based on the mode mapping.

    Parameters
    ----------
    factors1_dict : dict
        Dictionary with integer keys (dimension indices) and numpy array values
        (factor loadings) for tensor1.

    factors2_dict : dict
        Dictionary with integer keys (dimension indices) and numpy array values
        (factor loadings) for tensor2.

    mode_mapping : dict
        Mode mapping specification: {'shared': [(t1_mode, t2_mode), ...]}

    Returns
    -------
    combined_dict : dict
        Combined dictionary with all factor loadings. Shared modes use tensor1's
        version, and tensor-specific modes are added with unique keys.
    '''
    combined_dict = {}
    shared_pairs = mode_mapping.get('shared', [])

    # Get shared modes
    shared_t1_modes = set([pair[0] for pair in shared_pairs])
    shared_t2_modes = set([pair[1] for pair in shared_pairs])

    # Identify tensor-specific modes
    tensor1_only = [i for i in factors1_dict.keys() if i not in shared_t1_modes]
    tensor2_only = [i for i in factors2_dict.keys() if i not in shared_t2_modes]

    # Add shared factors (use tensor1's version)
    current_idx = 0
    for t1_mode, t2_mode in shared_pairs:
        combined_dict[current_idx] = factors1_dict[t1_mode]
        current_idx += 1

    # Add tensor1-specific factors
    for mode in tensor1_only:
        combined_dict[current_idx] = factors1_dict[mode]
        current_idx += 1

    # Add tensor2-specific factors
    for mode in tensor2_only:
        combined_dict[current_idx] = factors2_dict[mode]
        current_idx += 1

    return combined_dict


def _multiple_runs_coupled_elbow_analysis(tensor1, tensor2, mode_mapping, upper_rank=50, runs=10,
                                          tf_type='coupled_non_negative_cp', init='svd', svd='truncated_svd',
                                          metric='error', random_state=None, mask1=None, mask2=None,
                                          n_iter_max=100, tol=10e-7, verbose=False, balance_errors=True,
                                          manual_weights=(0.5, 0.5), **kwargs):
    '''
    Performs a coupled elbow analysis with multiple runs and mode mapping

    Parameters
    ----------
    tensor1 : tensorly.tensor
        First tensor to factorize.

    tensor2 : tensorly.tensor
        Second tensor to factorize.

    mode_mapping : dict or int/list (for backward compatibility)
        Mode mapping specification. Can be:
        - dict: {'shared': [(t1_mode, t2_mode), ...]}     # Pairs of shared modes
        - int/list: non_shared_modes (for backward compatibility with same-dimension tensors)

    upper_rank : int, default=50
        Maximum rank to evaluate.

    runs : int, default=10
        Number of tensor factorization performed for a given rank. Each factorization
        varies in the seed of initialization.

    tf_type : str, default='coupled_non_negative_cp'
        Type of Tensor Factorization.

    init : str, default='svd'
        Initialization method for computing the Tensor Factorization.
        {'svd', 'random'}

    svd : str, default='truncated_svd'
        Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS

    metric : str, default='error'
        Metric to perform the elbow analysis (y-axis)

        - 'error' : Normalized error to compute the elbow.
        - 'similarity' : Similarity based on CorrIndex (1-CorrIndex).

    random_state : int, default=None
        Seed for randomization.

    mask1 : tensorly.tensor, default=None
        Mask for the first tensor.

    mask2 : tensorly.tensor, default=None
        Mask for the second tensor.

    n_iter_max : int, default=100
        Maximum number of iterations.

    tol : float, default=10e-7
        Convergence tolerance.

    balance_errors : boolean, default=True
        Whether to balance the errors from each tensor based on their sizes.

    manual_weights : tuple, default=(0.5, 0.5)
        Manual weights (weight1, weight2) for importance of tensors in the factorization.
        Weights should be positive. Example: (2.0, 1.0) gives tensor1 twice
        the importance of tensor2 in both the factorization and the combined error metric.
        If None, automatic weight calculation is performed to have weigh1 and weight2
        inversely proportional to non-shared mode dimensions of each tensor.

    verbose : boolean, default=False
        Whether printing or not steps of the analysis.

    **kwargs : dict
        Extra arguments for the tensor factorization.

    Returns
    -------
    all_loss : dict
        Dictionary with keys 'tensor1', 'tensor2', and 'combined', each containing
        arrays of shape (runs, upper_rank) with the metric values (errors or similarities).
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

            cp1, cp2, (errors1, errors2) = _compute_coupled_tensor_factorization(
                tensor1=tensor1,
                tensor2=tensor2,
                rank=r,
                mode_mapping=mode_mapping,
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
                manual_weights=manual_weights,
                **kwargs
            )

            if metric == 'error':
                # Calculate individual and combined errors
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
                weight1, weight2 = _compute_balancing_weights(tensor1, tensor2, mode_mapping, balance_errors,
                                                              manual_weights)
                combined_error = (weight1 * error1 + weight2 * error2) / (weight1 + weight2)

                run_errors.append({
                    'tensor1': error1,
                    'tensor2': error2,
                    'combined': combined_error
                })

            elif metric == 'similarity':
                # Store factors from both tensors
                (weights1, factors1) = cp1
                (weights2, factors2) = cp2

                factors1_dict = dict(zip(list(range(len(factors1))), [tl.to_numpy(f) for f in factors1]))
                factors2_dict = dict(zip(list(range(len(factors2))), [tl.to_numpy(f) for f in factors2]))

                # Create combined factors dictionary
                combined_factors_dict = _create_combined_factors_dict(factors1_dict, factors2_dict, mode_mapping)

                run_errors.append({
                    'tensor1': factors1_dict,
                    'tensor2': factors2_dict,
                    'combined': combined_factors_dict
                })

        if metric == 'similarity':
            # Compute pairwise correlation index for each set of factors
            corridx_t1 = pairwise_correlation_index([d['tensor1'] for d in run_errors])
            corridx_t2 = pairwise_correlation_index([d['tensor2'] for d in run_errors])
            corridx_combined = pairwise_correlation_index([d['combined'] for d in run_errors])

            # Convert to distance metric (1 - similarity)
            similarity_t1 = 1.0 - sp.distance.squareform(corridx_t1.values)
            similarity_t2 = 1.0 - sp.distance.squareform(corridx_t2.values)
            similarity_combined = 1.0 - sp.distance.squareform(corridx_combined.values)

            run_errors = {
                'tensor1': similarity_t1.tolist(),
                'tensor2': similarity_t2.tolist(),
                'combined': similarity_combined.tolist()
            }
        elif metric == 'error':
            # Reorganize error data
            run_errors = {
                'tensor1': [d['tensor1'] for d in run_errors],
                'tensor2': [d['tensor2'] for d in run_errors],
                'combined': [d['combined'] for d in run_errors]
            }

        all_loss.append(run_errors)

    # Reorganize into separate arrays for each tensor
    all_loss = {
        'tensor1': np.array([all_loss[i]['tensor1'] for i in range(len(all_loss))]).T,
        'tensor2': np.array([all_loss[i]['tensor2'] for i in range(len(all_loss))]).T,
        'combined': np.array([all_loss[i]['combined'] for i in range(len(all_loss))]).T
    }

    return all_loss