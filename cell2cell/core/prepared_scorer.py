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

    row_cells, col_cells : list, default=None
        Cells to put on the rows and on the columns of the returned matrices. Both
        None, the default, gives the full square matrix over every cell in the
        interaction space.

        Naming them separately computes only that block, which is worth doing when
        the question is about one group of cells against another -- a set of cell
        types against the rest, say. The saving is real: the precomputed outer
        products, and the matrix product that uses them, go from `cells^2` to
        `len(row_cells) * len(col_cells)`.

        The block is the corresponding block of the full matrix, not an
        approximation of it. For `cci_type='undirected'` that holds **provided both
        directions of each ligand-receptor pair carry the same weight**, which is
        what a bidirectional PPI table scored with per-interaction weights gives; the
        full-matrix path can symmetrize after the fact, and a block cannot.

    Attributes
    ----------
    A, B : numpy.ndarray
        Weighted ligand and receptor expression, of shape (PPIs, cells). Over every
        cell of the interaction space, whatever `row_cells`/`col_cells` ask for.

    cell_names : list
        Cell names, in the order they have in the interaction space.

    row_names, col_names : list
        Cells on each axis of the returned matrices. Equal to `cell_names` unless
        `row_cells`/`col_cells` were given.

    square : boolean
        Whether the two axes hold the same cells in the same order.

    self_pairs : numpy.ndarray
        Boolean (rows, columns) array, True where the same cell is on both axes.
        These are the self-interactions; with different row and column sets they are
        not the matrix diagonal.

    batched : boolean
        Whether the precomputed outer products fitted in `max_memory_mb`.
    '''

    def __init__(self, interaction_space, cci_score=None, max_memory_mb=512,
                 row_cells=None, col_cells=None):
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

        # Row and column cell subsets. Both None is the full square matrix, which is
        # what every caller that does not ask for a block gets.
        known = set(self.cell_names)
        self.row_names = list(self.cell_names) if row_cells is None else list(row_cells)
        self.col_names = list(self.cell_names) if col_cells is None else list(col_cells)
        for label, subset in (('row_cells', self.row_names), ('col_cells', self.col_names)):
            if len(set(subset)) != len(subset):
                raise ValueError('`{}` lists the same cell more than once, which would '
                                 'weight it twice.'.format(label))
            missing = [cell for cell in subset if cell not in known]
            if missing:
                raise ValueError('`{}` has cells that are not in the interaction space: {}'
                                 .format(label, missing[:5]))
        position = {cell: i for i, cell in enumerate(self.cell_names)}
        row_idx = [position[cell] for cell in self.row_names]
        col_idx = [position[cell] for cell in self.col_names]
        self.n_rows, self.n_cols = len(row_idx), len(col_idx)
        self.square = self.row_names == self.col_names

        self._A_rows = self.A[:, row_idx]
        self._B_cols = self.B[:, col_idx]
        self._A2 = self._A_rows * self._A_rows
        self._B2 = self._B_cols * self._B_cols

        # Where the same cell sits on both axes. With different row and column sets
        # these are not the matrix diagonal, so they are found by name rather than by
        # position.
        self.self_pairs = (np.asarray(self.row_names, dtype=object)[:, None] ==
                           np.asarray(self.col_names, dtype=object)[None, :])

        # 'count' asks whether a product is non-zero, not how large it is, and
        # `[w * A * B != 0]` factorizes into the three indicators. Keeping them lets
        # the per-vector path count as well, rather than adding the products up.
        if self.cci_score == 'count':
            self._A_nonzero = (self._A_rows != 0).astype(float)
            self._B_nonzero = (self._B_cols != 0).astype(float)

        # Outer product of every ligand-receptor pair, flattened over the two cell
        # axes so the contraction over PPIs is a plain matrix product.
        outer_mb = self.n_ppi * self.n_rows * self.n_cols * 8 / 1e6
        self.batched = outer_mb <= max_memory_mb
        if self.batched:
            outer = self._A_rows[:, :, None] * self._B_cols[:, None, :]
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
            N = (W @ self._P).reshape(-1, self.n_rows, self.n_cols)
        elif self.cci_score == 'count':
            # Same indicators as the batched `_P`, so the result does not depend on
            # whether the outer products happened to fit in `max_memory_mb`
            A, B = self._A_nonzero, self._B_nonzero
            N = np.stack([(A * (w != 0)[:, None]).T @ B for w in W])
        else:
            N = np.stack([(self._A_rows * w[:, None]).T @ self._B_cols for w in W])
        return N, SA, SB

    def _combine(self, N, SA, SB):
        '''Applies the score-specific formula. Shapes: N (n, R, S), SA (n, R), SB (n, S).

        Already shape-agnostic: the broadcast below spans a rectangular block just as
        it spans a square one.
        '''
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
            CCI matrices, of shape (n, rows, columns), following `self.row_names`
            and `self.col_names`. For an undirected interaction space over the full
            square matrix these are symmetrized the same way
            `compute_pairwise_cci_scores` does, by mirroring the upper triangle. A
            rectangular block cannot be mirrored, and does not need to be: with both
            directions of each interaction equally weighted the result is already
            symmetric, so the block equals the same block of the full matrix.
        '''
        W = np.atleast_2d(np.asarray(W, dtype=float))
        if W.shape[1] != self.n_ppi:
            raise ValueError('Expected weight vectors of length {}, got {}'
                             .format(self.n_ppi, W.shape[1]))

        # Checked on both paths, so the same weights are accepted or rejected whatever
        # `max_memory_mb` decided
        if self.cci_score == 'count' and not np.isin(W, (0.0, 1.0)).all():
            raise ValueError("The 'count' score is only vectorized for binary weights, "
                             "because it counts non-zero products rather than adding them.")

        scores = self._combine(*self._terms(W))

        if self.cci_type == 'undirected' and self.square:
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
        return pd.DataFrame(scores, index=self.row_names, columns=self.col_names)

    def distance_batch(self, W, zero_diagonal=True):
        '''
        Computes the distance matrix for each of several PPI weight vectors,
        reproducing what `InteractionSpace.compute_pairwise_cci_scores` derives.

        Bounded scores use `1 - score`; the unbounded ones ('count', 'icellnet')
        use the regularized `1 - score / (score + mean)`, where the mean is taken
        over the CCI matrix of that weight vector.

        Note that for the unbounded scores over a rectangular block that mean is
        taken over the block rather than over the full matrix, so the distances
        differ from the full-matrix ones by a per-candidate constant. A rank
        correlation is unaffected, since the transform stays monotone; a Pearson
        one is not.

        Parameters
        ----------
        W : numpy.ndarray
            Weights, of shape (n, PPIs).

        zero_diagonal : boolean, default=True
            Whether to zero the self-interactions, as
            `compute_pairwise_cci_scores` does. Pass False to keep the autocrine
            distances, which is only useful when they are going to be read.

        Returns
        -------
        distances : numpy.ndarray
            Distance matrices, of shape (n, rows, columns).
        '''
        scores = self.score_batch(W)
        if self.cci_score in UNBOUNDED_CCI_SCORES:
            means = np.nanmean(scores, axis=(1, 2))[:, None, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                distances = 1.0 - np.divide(scores, scores + means)
        else:
            distances = 1.0 - scores

        if zero_diagonal:
            distances[:, self.self_pairs] = 0.0
        return distances
