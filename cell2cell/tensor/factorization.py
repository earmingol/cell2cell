# -*- coding: utf-8 -*-

import numpy as np
from tqdm.auto import tqdm
from kneed import KneeLocator

from tensorly.decomposition import non_negative_parafac


def _compute_tensor_factorization(tensor, rank, tf_type='non_negative_cp', init='svd', random_state=None, mask=None,
                                  verbose=False, **kwargs):
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
    kneedle = KneeLocator(*zip(*loss), S=1.0, curve="convex", direction="decreasing")
    rank = kneedle.elbow
    return rank


def _run_elbow_analysis(tensor, upper_rank=50, tf_type='non_negative_cp', init='svd', random_state=None, mask=None,
                        verbose=False, disable_pbar=False, **kwargs):
    loss = []
    for r in tqdm(range(1, upper_rank + 1), disable=disable_pbar):
        tl_object = _compute_tensor_factorization(tensor=tensor,
                                                  rank=r,
                                                  tf_type=tf_type,
                                                  init=init,
                                                  random_state=random_state,
                                                  mask=mask,
                                                  verbose=verbose,
                                                  **kwargs)
        loss.append((r, _compute_norm_error(tensor, tl_object, mask)))

    return loss


def _multiple_runs_elbow_analysis(tensor, upper_rank=50, runs=10, tf_type='non_negative_cp', init='svd', random_state=None,
                                  mask=None, verbose=False, **kwargs):
    assert isinstance(runs, int), "runs must be an integer"
    all_loss = []
    for r in tqdm(range(1, upper_rank + 1)):
        run_errors = []
        for run in range(runs):
            if random_state is not None:
                rs = random_state + run
            else:
                rs = None
            tl_object = _compute_tensor_factorization(tensor=tensor,
                                                      rank=r,
                                                      tf_type=tf_type,
                                                      init=init,
                                                      random_state=rs,
                                                      mask=mask,
                                                      verbose=verbose,
                                                      **kwargs)
            run_errors.append(_compute_norm_error(tensor, tl_object, mask))
        all_loss.append(run_errors)

    all_loss = np.array(all_loss).T
    return all_loss


def _compute_norm_error(tensor, tl_object, mask=None):
    assert tl_object is not None, "Before computing the error, please run the function _compute_tensor_factorization to generate the tl_object"

    # Compute error
    rec_ = tl_object.to_tensor()
    if mask is not None:
        tensor_ = tensor * mask + rec_ * (1 - mask)
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
    return np.linalg.norm(reference_tensor - reconstructed_tensor) / np.linalg.norm(reference_tensor)