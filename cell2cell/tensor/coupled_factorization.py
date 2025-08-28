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


def _process_mode_mapping(tensor1, tensor2, mode_mapping):
    """Process mode mapping input, handling backward compatibility"""

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
    """Validate that the two tensors are compatible for coupled factorization

    Parameters
    ----------
    tensor1, tensor2 : tensor objects or raw tensors
        Can be either tensorly tensors or objects with .tensor attribute
    """
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

    mode_mapping : dict
        Dictionary specifying shared mode relationships:
        {
            'shared': [(t1_mode, t2_mode), ...]     # Pairs of shared modes
        }
        Tensor-specific modes are automatically derived.

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

    # Calculate balancing weights based on tensor-specific dimensions
    if balance_errors:
        nonshared_size1 = np.prod([tensor1.shape[i] for i in tensor1_only]) if tensor1_only else 1
        nonshared_size2 = np.prod([tensor2.shape[i] for i in tensor2_only]) if tensor2_only else 1
        total_nonshared = nonshared_size1 + nonshared_size2
        if total_nonshared > 0:
            weight1 = total_nonshared / nonshared_size1 if nonshared_size1 > 0 else 1.0
            weight2 = total_nonshared / nonshared_size2 if nonshared_size2 > 0 else 1.0
        else:
            weight1 = weight2 = 1.0
    else:
        weight1 = weight2 = 1.0

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
                numerator = tl.clip((mttkrp1 + mttkrp2) / 2., a_min=epsilon, a_max=None)
                denominator = tl.clip(
                    (tl.dot(factors1[t1_mode], accum1) + tl.dot(factors2[t2_mode], accum2)) / 2.,
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


def _compute_coupled_tensor_factorization(tensor1, tensor2, rank, mode_mapping, mask1=None, mask2=None,
                                          tf_type='coupled_non_negative_cp', init='svd', svd='truncated_svd',
                                          random_state=None, n_iter_max=100, tol=10e-7, verbose=False,
                                          balance_errors=True, **kwargs):
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
            **kwargs
        )
    else:
        raise ValueError('Not a valid tf_type for coupled factorization.')

    return result


def _run_coupled_elbow_analysis(tensor1, tensor2, mode_mapping, upper_rank=50, tf_type='coupled_non_negative_cp',
                                init='svd', svd='truncated_svd', random_state=None, mask1=None, mask2=None,
                                n_iter_max=100, tol=10e-7, verbose=False, balance_errors=True,
                                disable_pbar=False, **kwargs):
    '''Performs a coupled elbow analysis with mode mapping'''

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
            **kwargs
        )

        # Calculate combined error
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

        # Calculate combined error with balancing based on non-shared dimensions
        if balance_errors:
            # Automatically derive tensor-specific modes
            shared_t1_modes = set([pair[0] for pair in mode_mapping.get('shared', [])])
            shared_t2_modes = set([pair[1] for pair in mode_mapping.get('shared', [])])
            tensor1_only = [i for i in range(tl.ndim(tensor1)) if i not in shared_t1_modes]
            tensor2_only = [i for i in range(tl.ndim(tensor2)) if i not in shared_t2_modes]

            nonshared_size1 = np.prod([tensor1.shape[i] for i in tensor1_only]) if tensor1_only else 1
            nonshared_size2 = np.prod([tensor2.shape[i] for i in tensor2_only]) if tensor2_only else 1
            total_nonshared = nonshared_size1 + nonshared_size2
            if total_nonshared > 0:
                weight1 = total_nonshared / nonshared_size1 if nonshared_size1 > 0 else 1.0
                weight2 = total_nonshared / nonshared_size2 if nonshared_size2 > 0 else 1.0
                combined_error = (weight1 * error1 + weight2 * error2) / (weight1 + weight2)
            else:
                combined_error = (error1 + error2) / 2
        else:
            combined_error = (error1 + error2) / 2

        loss.append((r, combined_error))

    return loss


def _multiple_runs_coupled_elbow_analysis(tensor1, tensor2, mode_mapping, upper_rank=50, runs=10,
                                          tf_type='coupled_non_negative_cp', init='svd', svd='truncated_svd',
                                          metric='error', random_state=None, mask1=None, mask2=None,
                                          n_iter_max=100, tol=10e-7, verbose=False, balance_errors=True, **kwargs):
    '''Performs a coupled elbow analysis with multiple runs and mode mapping'''

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
                **kwargs
            )

            # Calculate combined error
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
            if balance_errors:
                # Automatically derive tensor-specific modes
                shared_t1_modes = set([pair[0] for pair in mode_mapping.get('shared', [])])
                shared_t2_modes = set([pair[1] for pair in mode_mapping.get('shared', [])])
                tensor1_only = [i for i in range(tl.ndim(tensor1)) if i not in shared_t1_modes]
                tensor2_only = [i for i in range(tl.ndim(tensor2)) if i not in shared_t2_modes]

                nonshared_size1 = np.prod([tensor1.shape[i] for i in tensor1_only]) if tensor1_only else 1
                nonshared_size2 = np.prod([tensor2.shape[i] for i in tensor2_only]) if tensor2_only else 1
                total_nonshared = nonshared_size1 + nonshared_size2
                if total_nonshared > 0:
                    weight1 = total_nonshared / nonshared_size1 if nonshared_size1 > 0 else 1.0
                    weight2 = total_nonshared / nonshared_size2 if nonshared_size2 > 0 else 1.0
                    combined_error = (weight1 * error1 + weight2 * error2) / (weight1 + weight2)
                else:
                    combined_error = (error1 + error2) / 2
            else:
                combined_error = (error1 + error2) / 2

            run_errors.append(combined_error)

        all_loss.append(run_errors)

    all_loss = np.array(all_loss).T
    return all_loss