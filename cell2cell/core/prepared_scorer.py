# -*- coding: utf-8 -*-

'''Vectorized cell-cell interaction scoring under varying PPI weights.

Scoring the same cells again with a different set of ligand-receptor weights is a
common inner loop -- optimization searches, permutation nulls, knockout scans. Doing
it by rebuilding an `InteractionSpace` each time is orders of magnitude slower than
it needs to be, because the expression-derived part does not change.
'''

from __future__ import absolute_import

import numpy as np
import pandas as pd


LINEAR_CCI_SCORES = ('bray_curtis', 'jaccard', 'count', 'icellnet')

# Scores that are not bounded between 0 and 1, and whose distance matrix is
# therefore computed with the regularized formula in `InteractionSpace`.
UNBOUNDED_CCI_SCORES = ('count', 'icellnet')


class PreparedCCIScorer:
    '''
    Precomputes the expression-dependent part of the CCI scores, so that scoring
    the same cells again under a different set of PPI weights becomes a matrix
    multiplication instead of a rebuild of the interaction space.

    Every CCI score in `cell2cell.core.cci_scores` is a function of three
    quantities, each of which is a weighted sum over the PPIs and therefore
    **linear in the PPI weights** `w`:

    .. code-block:: text

        N(i, j)  = sum_k w_k * A_ki * B_kj      (the ligand-receptor product)
        SA(i)    = sum_k w_k * A_ki^2
        SB(j)    = sum_k w_k * B_kj^2

        bray_curtis = 2 * N / (SA + SB)
        jaccard     =     N / (SA + SB - N)
        icellnet    =     N
        count       = sum_k [w_k * A_ki * B_kj != 0]

    Since `A` and `B` depend only on the expression data, they can be computed
    once and reused for any number of weight vectors. Scoring a whole population
    of weight vectors is then a single matrix product.

    Parameters
    ----------
    interaction_space : cell2cell.core.interaction_space.InteractionSpace
        A built interaction space. Only its expression-derived matrices are used;
        it is not modified.

    cci_score : str, default=None
        CCI score to compute. If None, the one the interaction space was built
        with is used. Must be one of 'bray_curtis', 'jaccard', 'count' or
        'icellnet'.

    max_memory_mb : float, default=512
        Budget for the precomputed ligand-receptor outer products, which are what
        make the batched path possible. If the array would exceed this, only the
        per-vector path is prepared, which is still much faster than rebuilding
        the interaction space but does not gain from batching.

    Attributes
    ----------
    A, B : numpy.ndarray
        Weighted ligand and receptor expression, of shape (PPIs, cells).

    cell_names : list
        Cell names, in the order they have in the interaction space. Rows and
        columns of every returned matrix follow this order.

    batched : boolean
        Whether the precomputed outer products fitted in `max_memory_mb`.
    '''

    def __init__(self, interaction_space, cci_score=None, max_memory_mb=512):
        if cci_score is None:
            cci_score = interaction_space.cci_score
        if cci_score not in LINEAR_CCI_SCORES:
            raise NotImplementedError(
                "'{}' is not supported by the vectorized scorer. Use one of {}, or "
                "pass fast=False to fall back to the reference implementation."
                .format(cci_score, list(LINEAR_CCI_SCORES)))

        self.cci_score = cci_score
        self.cci_type = interaction_space.cci_type
        self.cell_names = list(interaction_space.interaction_elements['cell_names'])

        cells = interaction_space.interaction_elements['cells']
        self.A = np.column_stack([cells[c].weighted_ppi['A'].values for c in self.cell_names])
        self.B = np.column_stack([cells[c].weighted_ppi['B'].values for c in self.cell_names])

        # `nansum` in the scalar scores treats missing values as zero. Doing the
        # substitution once here reproduces that without a NaN-aware reduction in
        # the inner loop.
        self._has_nans = bool(np.isnan(self.A).any() or np.isnan(self.B).any())
        self.A = np.nan_to_num(self.A)
        self.B = np.nan_to_num(self.B)

        self.n_ppi, self.n_cells = self.A.shape
        self._A2 = self.A * self.A
        self._B2 = self.B * self.B

        # Outer product of every ligand-receptor pair, flattened over the two cell
        # axes so the contraction over PPIs is a plain matrix product.
        outer_mb = self.n_ppi * self.n_cells * self.n_cells * 8 / 1e6
        self.batched = outer_mb <= max_memory_mb
        if self.batched:
            outer = self.A[:, :, None] * self.B[:, None, :]
            if self.cci_score == 'count':
                # 'count' counts the non-zero products rather than adding them up,
                # which is linear in a binary weight vector but not in a general one.
                self._P = (outer != 0).astype(float).reshape(self.n_ppi, -1)
            else:
                self._P = outer.reshape(self.n_ppi, -1)
        else:
            self._P = None

    def _terms(self, W):
        '''Computes N, SA and SB for a stack of weight vectors W of shape (n, PPIs).'''
        SA = W @ self._A2
        SB = W @ self._B2
        if self.batched:
            N = (W @ self._P).reshape(-1, self.n_cells, self.n_cells)
        else:
            N = np.stack([(self.A * w[:, None]).T @ self.B for w in W])
        return N, SA, SB

    def _combine(self, N, SA, SB):
        '''Applies the score-specific formula. Shapes: N (n, C, C), SA/SB (n, C).'''
        if self.cci_score == 'icellnet' or self.cci_score == 'count':
            return N

        denominator = SA[:, :, None] + SB[:, None, :]
        if self.cci_score == 'jaccard':
            denominator = denominator - N

        with np.errstate(divide='ignore', invalid='ignore'):
            if self.cci_score == 'bray_curtis':
                scores = np.divide(2.0 * N, denominator)
            else:
                scores = np.divide(N, denominator)
        # The scalar implementations return 0.0 when the denominator is zero
        scores[denominator == 0.0] = 0.0
        return scores

    def score_batch(self, W):
        '''
        Computes the CCI matrix for each of several PPI weight vectors.

        Parameters
        ----------
        W : numpy.ndarray
            Weights, of shape (n, PPIs). One row per weight vector.

        Returns
        -------
        scores : numpy.ndarray
            CCI matrices, of shape (n, cells, cells). Rows and columns follow
            `self.cell_names`. For an undirected interaction space the matrices
            are symmetrized the same way `compute_pairwise_cci_scores` does, by
            mirroring the upper triangle.
        '''
        W = np.atleast_2d(np.asarray(W, dtype=float))
        if W.shape[1] != self.n_ppi:
            raise ValueError('Expected weight vectors of length {}, got {}'
                             .format(self.n_ppi, W.shape[1]))

        if self.batched and self.cci_score == 'count' and not np.isin(W, (0.0, 1.0)).all():
            raise ValueError("The 'count' score is only vectorized for binary weights, "
                             "because it counts non-zero products rather than adding them.")

        scores = self._combine(*self._terms(W))

        if self.cci_type == 'undirected':
            # `generate_pairs` yields the upper triangle plus the diagonal, and the
            # scoring loop mirrors each value. The lower triangle of the directed
            # result is therefore never used.
            upper = np.triu(scores)
            scores = upper + np.triu(scores, k=1).transpose(0, 2, 1)
        return scores

    def score(self, ppi_score):
        '''
        Computes the CCI matrix for a single PPI weight vector.

        Parameters
        ----------
        ppi_score : array-like
            Weights, one per PPI.

        Returns
        -------
        cci_matrix : pandas.DataFrame
            CCI scores, with cells as rows and columns.
        '''
        scores = self.score_batch(np.asarray(ppi_score, dtype=float)[None, :])[0]
        return pd.DataFrame(scores, index=self.cell_names, columns=self.cell_names)

    def distance_batch(self, W):
        '''
        Computes the distance matrix for each of several PPI weight vectors,
        reproducing what `InteractionSpace.compute_pairwise_cci_scores` derives.

        Bounded scores use `1 - score`; the unbounded ones ('count', 'icellnet')
        use the regularized `1 - score / (score + mean)`, where the mean is taken
        over the whole CCI matrix of that weight vector. The diagonal is zeroed.

        Parameters
        ----------
        W : numpy.ndarray
            Weights, of shape (n, PPIs).

        Returns
        -------
        distances : numpy.ndarray
            Distance matrices, of shape (n, cells, cells).
        '''
        scores = self.score_batch(W)
        if self.cci_score in UNBOUNDED_CCI_SCORES:
            means = np.nanmean(scores, axis=(1, 2))[:, None, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                distances = 1.0 - np.divide(scores, scores + means)
        else:
            distances = 1.0 - scores

        idx = np.arange(self.n_cells)
        distances[:, idx, idx] = 0.0
        return distances
