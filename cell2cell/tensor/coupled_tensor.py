# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorly as tl

from collections import OrderedDict
from tqdm import tqdm

from cell2cell.tensor.tensor import BaseTensor
from cell2cell.tensor.factorization import _compute_elbow
from cell2cell.plotting.tensor_plot import plot_elbow, plot_multiple_run_elbow
from cell2cell.preprocessing.signal import smooth_curve
from cell2cell.tensor.coupled_factorization import coupled_non_negative_parafac


class CoupledInteractionTensor(BaseTensor):
    '''
    Coupled Tensor Factorization for two interaction tensors that share all but one dimension.

    This class performs simultaneous non-negative CP decomposition on two tensors that share
    all dimensions except one. The shared dimensions will have the same factor matrices,
    while the non-shared dimension will have separate factor matrices.

    Parameters
    ----------
    tensor1 : cell2cell.tensor.BaseTensor
        First interaction tensor (e.g., InteractionTensor, PreBuiltTensor)

    tensor2 : cell2cell.tensor.BaseTensor
        Second interaction tensor (e.g., InteractionTensor, PreBuiltTensor)

    non_shared_mode : int
        The mode (dimension) that differs between the two tensors

    tensor1_name : str, default='Tensor1'
        Name for the first tensor (used in factor labeling)

    tensor2_name : str, default='Tensor2'
        Name for the second tensor (used in factor labeling)

    balance_errors : bool, default=True
        Whether to balance the errors based on the size of non-shared dimensions

    device : str, default=None
        Device to use when backend allows using multiple devices

    Attributes
    ----------
    tensor1 : tensorly.tensor
        First tensor object

    tensor2 : tensorly.tensor
        Second tensor object

    non_shared_mode : int
        The dimension that differs between tensors

    cp1 : CPTensor
        CP decomposition result for tensor1

    cp2 : CPTensor
        CP decomposition result for tensor2

    factors1 : dict
        Factor loadings for tensor1

    factors2 : dict
        Factor loadings for tensor2

    factors : dict
        Combined factor loadings with shared and non-shared factors

    order_names1 : list
        Element names for each dimension of tensor1

    order_names2 : list
        Element names for each dimension of tensor2

    order_labels1 : list
        Dimension labels for tensor1

    order_labels2 : list
        Dimension labels for tensor2
    '''

    def __init__(self, tensor1, tensor2, non_shared_mode, tensor1_name='Tensor1',
                 tensor2_name='Tensor2', balance_errors=True, device=None):
        # Initialize BaseTensor
        BaseTensor.__init__(self)

        # Validate inputs
        self._validate_tensors(tensor1, tensor2, non_shared_mode)

        # Store tensor objects and metadata
        self.tensor1 = tensor1.tensor
        self.tensor2 = tensor2.tensor
        self.non_shared_mode = non_shared_mode
        self.tensor1_name = tensor1_name
        self.tensor2_name = tensor2_name
        self.balance_errors = balance_errors

        # Store order information
        self.order_names1 = tensor1.order_names.copy()
        self.order_names2 = tensor2.order_names.copy()
        self.order_labels1 = tensor1.order_labels.copy() if tensor1.order_labels else None
        self.order_labels2 = tensor2.order_labels.copy() if tensor2.order_labels else None

        # Store masks if available
        self.mask1 = tensor1.mask
        self.mask2 = tensor2.mask

        # Initialize factorization results
        self.tl_object1 = None
        self.tl_object2 = None
        self.norm_tl_object1 = None
        self.norm_tl_object2 = None
        self.factors1 = None
        self.factors2 = None
        self.factors = None

        # Move to device if specified
        if device is not None:
            self.to_device(device)

    def _validate_tensors(self, tensor1, tensor2, non_shared_mode):
        """Validate that the two tensors are compatible for coupled factorization"""
        if tl.ndim(tensor1.tensor) != tl.ndim(tensor2.tensor):
            raise ValueError("Both tensors must have the same number of dimensions")

        n_modes = tl.ndim(tensor1.tensor)
        if non_shared_mode >= n_modes or non_shared_mode < 0:
            raise ValueError(f"non_shared_mode must be between 0 and {n_modes - 1}")

        # Check that all other dimensions match
        for mode in range(n_modes):
            if mode != non_shared_mode:
                if tensor1.tensor.shape[mode] != tensor2.tensor.shape[mode]:
                    raise ValueError(f"Tensors must have the same size in mode {mode}")

                # Check that element names match for shared dimensions
                if tensor1.order_names[mode] != tensor2.order_names[mode]:
                    raise ValueError(f"Element names must match for shared dimension {mode}")

    def compute_tensor_factorization(self, rank, n_iter_max=100, init='svd',
                                     svd='truncated_svd', tol=10e-7, random_state=None,
                                     normalize_loadings=True, var_ordered_factors=True,
                                     cvg_criterion='abs_rec_error', verbose=False,
                                     runs=1, **kwargs):
        '''
        Performs coupled tensor factorization on both tensors

        Parameters
        ----------
        rank : int
            Number of components for the factorization

        n_iter_max : int, default=100
            Maximum number of iterations

        init : str, default='svd'
            Initialization method {'svd', 'random'}

        svd : str, default='truncated_svd'
            SVD function to use

        tol : float, default=1e-7
            Convergence tolerance

        random_state : int, default=None
            Random state for reproducibility

        normalize_loadings : bool, default=True
            Whether to normalize factors

        var_ordered_factors : bool, default=True
            Whether to order factors by variance explained

        cvg_criterion : str, default='abs_rec_error'
            Convergence criterion {'abs_rec_error', 'rec_error'}

        verbose : bool, default=False
            Whether to print progress

        runs : int, default=1
            Number of runs to find best solution

        **kwargs : dict
            Additional arguments for the factorization
        '''

        best_error = np.inf
        best_cp1, best_cp2 = None, None

        for run in tqdm(range(runs), disable=(runs == 1)):
            if random_state is not None:
                rs = random_state + run
            else:
                rs = None

            # Perform coupled factorization
            if 'return_errors' in kwargs and kwargs['return_errors']:
                cp1, cp2, (errors1, errors2) = coupled_non_negative_parafac(
                    tensor1=self.tensor1,
                    tensor2=self.tensor2,
                    rank=rank,
                    non_shared_mode=self.non_shared_mode,
                    n_iter_max=n_iter_max,
                    init=init,
                    svd=svd,
                    tol=tol,
                    random_state=rs,
                    verbose=verbose,
                    normalize_factors=False,  # We'll handle normalization later
                    return_errors=True,
                    cvg_criterion=cvg_criterion,
                    balance_errors=self.balance_errors
                )
                # Calculate combined error for comparison
                N1 = self.tensor1.shape[self.non_shared_mode]
                N2 = self.tensor2.shape[self.non_shared_mode]
                if self.balance_errors:
                    total_N = N1 + N2
                    weight1 = total_N / N1
                    weight2 = total_N / N2
                    combined_error = (weight1 * errors1[-1] + weight2 * errors2[-1]) / (weight1 + weight2)
                else:
                    combined_error = (errors1[-1] + errors2[-1]) / 2

            else:
                cp1, cp2 = coupled_non_negative_parafac(
                    tensor1=self.tensor1,
                    tensor2=self.tensor2,
                    rank=rank,
                    non_shared_mode=self.non_shared_mode,
                    n_iter_max=n_iter_max,
                    init=init,
                    svd=svd,
                    tol=tol,
                    random_state=rs,
                    verbose=verbose,
                    normalize_factors=False,  # We'll handle normalization later
                    return_errors=False,
                    cvg_criterion=cvg_criterion,
                    balance_errors=self.balance_errors
                )
                # Calculate error for comparison if multiple runs
                if runs > 1:
                    from cell2cell.tensor.factorization import _compute_norm_error
                    error1 = _compute_norm_error(self.tensor1, cp1, self.mask1)
                    error2 = _compute_norm_error(self.tensor2, cp2, self.mask2)

                    # Apply balanced weighting if requested
                    N1 = self.tensor1.shape[self.non_shared_mode]
                    N2 = self.tensor2.shape[self.non_shared_mode]
                    if self.balance_errors:
                        total_N = N1 + N2
                        weight1 = total_N / N1
                        weight2 = total_N / N2
                        combined_error = (weight1 * error1 + weight2 * error2) / (weight1 + weight2)
                    else:
                        combined_error = (error1 + error2) / 2
                else:
                    combined_error = 0

            if runs > 1 and combined_error < best_error:
                best_error = combined_error
                best_cp1, best_cp2 = cp1, cp2
            elif runs == 1:
                best_cp1, best_cp2 = cp1, cp2

        if runs > 1:
            print(f'Best coupled model has a combined normalized error of: {best_error:.3f}')

        # Store results
        self.tl_object1 = best_cp1
        self.tl_object2 = best_cp2
        self.rank = rank

        # Create factor DataFrames
        self._create_factor_dataframes(normalize_loadings, var_ordered_factors)

        # Calculate explained variance
        self.explained_variance_ = self.explained_variance()

    def _create_factor_dataframes(self, normalize_loadings, var_ordered_factors):
        """Create factor DataFrames and unified factors"""

        factor_names = ['Factor {}'.format(i) for i in range(1, self.rank + 1)]

        # Get order labels
        if self.order_labels1 is None:
            tensor_dim = len(self.tensor1.shape)
            if tensor_dim == 4:
                self.order_labels1 = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
            elif tensor_dim > 4:
                self.order_labels1 = ['Contexts-{}'.format(i + 1) for i in range(tensor_dim - 3)] + [
                    'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
            elif tensor_dim == 3:
                self.order_labels1 = ['Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

        if self.order_labels2 is None:
            self.order_labels2 = self.order_labels1.copy()

        if normalize_loadings:
            self.norm_tl_object1 = tl.cp_tensor.cp_normalize(self.tl_object1)
            self.norm_tl_object2 = tl.cp_tensor.cp_normalize(self.tl_object2)
            # Extract factors
            (weights1, factors1) = self.norm_tl_object1
            (weights2, factors2) = self.norm_tl_object2
        else:
            # Extract factors
            (weights1, factors1) = self.tl_object1
            (weights2, factors2) = self.tl_object2

        # Handle normalization and ordering
        if normalize_loadings:
            weights1 = tl.to_numpy(weights1)
            weights2 = tl.to_numpy(weights2)

            if var_ordered_factors:
                # Use average weights for ordering
                avg_weights = (weights1 + weights2) / 2
                w_order = avg_weights.argsort()[::-1]
                factors1 = [tl.to_numpy(f)[:, w_order] for f in factors1]
                factors2 = [tl.to_numpy(f)[:, w_order] for f in factors2]
                self.explained_variance_ratio_ = avg_weights[w_order] / sum(avg_weights)
            else:
                factors1 = [tl.to_numpy(f) for f in factors1]
                factors2 = [tl.to_numpy(f) for f in factors2]
                avg_weights = (weights1 + weights2) / 2
                self.explained_variance_ratio_ = avg_weights / sum(avg_weights)
        else:
            factors1 = [tl.to_numpy(f) for f in factors1]
            factors2 = [tl.to_numpy(f) for f in factors2]
            self.explained_variance_ratio_ = None

        # Create factor DataFrames
        self.factors1 = OrderedDict(zip(self.order_labels1,
                                        [pd.DataFrame(f, index=idx, columns=factor_names)
                                         for f, idx in zip(factors1, self.order_names1)]))

        self.factors2 = OrderedDict(zip(self.order_labels2,
                                        [pd.DataFrame(f, index=idx, columns=factor_names)
                                         for f, idx in zip(factors2, self.order_names2)]))

        # Create unified factors with proper ordering
        unshared_tensor1_name = self.order_labels1[self.non_shared_mode]
        unshared_tensor2_name = self.order_labels2[self.non_shared_mode]

        # Only add suffix if the names match, otherwise keep original name
        if unshared_tensor1_name == unshared_tensor2_name:
            unified_key = f"{unshared_tensor2_name}_{self.tensor2_name}"
        else:
            unified_key = unshared_tensor2_name

        # Create ordered dictionary with tensor2 non-shared factor inserted
        # right after tensor1 non-shared factor
        self.factors = OrderedDict()

        for i, (key, value) in enumerate(self.factors1.items()):
            # Add the current factor from tensor1
            self.factors[key] = value

            # If this is the non-shared dimension, add tensor2's version right after
            if i == self.non_shared_mode:
                self.factors[unified_key] = self.factors2[unshared_tensor2_name].copy()

        # Store this for later use
        self._unshared_tensor2_key = unified_key

    def elbow_rank_selection(self, upper_rank=50, runs=20, tf_type='non_negative_cp', init='random', svd='truncated_svd',
                             metric='error', random_state=None, n_iter_max=100, tol=10e-7, automatic_elbow=True,
                             manual_elbow=None, smooth=False, ci='std', figsize=(4, 2.25), fontsize=14,
                             filename=None, output_fig=True, verbose=False, **kwargs):
        '''
        Elbow analysis for coupled tensor factorization

        Parameters
        ----------
        upper_rank : int, default=50
            Upper bound of ranks to explore

        runs : int, default=20
            Number of factorizations per rank

        tf_type : str, default='non_negative_cp'
            Type of tensor factorization (currently only supports 'non_negative_cp')

        init : str, default='random'
            Initialization method {'svd', 'random'}

        svd : str, default='truncated_svd'
            Function to compute the SVD, acceptable values in tensorly.SVD_FUNS

        metric : str, default='error'
            Metric for elbow analysis {'error', 'similarity'}

        random_state : int, default=None
            Random seed

        n_iter_max : int, default=100
            Maximum iterations per factorization

        tol : float, default=1e-7
            Convergence tolerance

        automatic_elbow : bool, default=True
            Whether to automatically detect elbow

        manual_elbow : int, default=None
            Manual elbow selection

        smooth : bool, default=False
            Whether to smooth the curve

        ci : str, default='std'
            Confidence interval type

        figsize : tuple, default=(4, 2.25)
            Figure size

        fontsize : int, default=14
            Font size for labels

        filename : str, default=None
            Path to save figure

        output_fig : bool, default=True
            Whether to generate figure

        verbose : bool, default=False
            Verbosity

        **kwargs : dict
            Additional arguments

        Returns
        -------
        fig : matplotlib.figure.Figure
            Elbow analysis figure

        loss : list
            List of (rank, error) tuples
        '''

        assert metric in ['similarity', 'error'], "`metric` must be either 'similarity' or 'error'"
        ylabel = {'similarity': 'Similarity\n(1-CorrIndex)', 'error': 'Normalized Error'}

        if verbose:
            print('Running Coupled Elbow Analysis')

        if metric == 'similarity':
            assert runs > 1, "`runs` must be greater than 1 when `metric` = 'similarity'"

        if kwargs is None:
            kwargs = {'return_errors': True}
        else:
            kwargs['return_errors'] = True

        # Run elbow analysis
        if runs == 1:
            loss = self._run_single_elbow_analysis(upper_rank, random_state, n_iter_max,
                                                   tol, verbose, **kwargs)
            all_loss = np.array([[l[1] for l in loss]])
        else:
            all_loss = self._run_multiple_elbow_analysis(upper_rank, runs, metric, random_state,
                                                         n_iter_max, tol, verbose, **kwargs)
            loss = np.nanmean(all_loss, axis=0).tolist()
            loss = [(i + 1, l) for i, l in enumerate(loss)]

        if smooth:
            loss_values = [l[1] for l in loss]
            smoothed_values = smooth_curve(loss_values)
            loss = [(i + 1, l) for i, l in enumerate(smoothed_values)]

        # Find elbow
        if automatic_elbow:
            rank = int(_compute_elbow(loss))
        else:
            rank = manual_elbow

        # Generate plot
        if output_fig:
            if runs == 1:
                fig = plot_elbow(loss=loss, elbow=rank, figsize=figsize,
                                 ylabel=ylabel[metric], fontsize=fontsize, filename=filename)
            else:
                fig = plot_multiple_run_elbow(all_loss=all_loss, ci=ci, elbow=rank,
                                              figsize=figsize, ylabel=ylabel[metric],
                                              smooth=smooth, fontsize=fontsize, filename=filename)
        else:
            fig = None

        # Store results
        self.rank = rank
        self.elbow_metric = metric
        self.elbow_metric_mean = loss
        self.elbow_metric_raw = all_loss

        if self.rank is not None:
            print(f'The rank at the elbow is: {self.rank}')

        return fig, loss

    def _run_single_elbow_analysis(self, upper_rank, random_state, n_iter_max, tol, verbose, **kwargs):
        """Run elbow analysis with single factorization per rank"""
        loss = []

        for r in tqdm(range(1, upper_rank + 1)):
            cp1, cp2, (errors1, errors2) = coupled_non_negative_parafac(
                tensor1=self.tensor1,
                tensor2=self.tensor2,
                rank=r,
                non_shared_mode=self.non_shared_mode,
                n_iter_max=n_iter_max,
                tol=tol,
                random_state=random_state,
                verbose=verbose,
                balance_errors=self.balance_errors,
                **kwargs
            )

            # Calculate combined error
            N1 = self.tensor1.shape[self.non_shared_mode]
            N2 = self.tensor2.shape[self.non_shared_mode]

            if self.balance_errors:
                total_N = N1 + N2
                weight1 = total_N / N1
                weight2 = total_N / N2
                combined_error = (weight1 * errors1[-1] + weight2 * errors2[-1]) / (weight1 + weight2)
            else:
                combined_error = (errors1[-1] + errors2[-1]) / 2

            loss.append((r, combined_error))

        return loss

    def _run_multiple_elbow_analysis(self, upper_rank, runs, metric, random_state,
                                     n_iter_max, tol, verbose, **kwargs):
        """Run elbow analysis with multiple factorizations per rank"""
        all_loss = []

        for r in tqdm(range(1, upper_rank + 1)):
            run_errors = []

            for run in range(runs):
                if random_state is not None:
                    rs = random_state + run
                else:
                    rs = None

                if metric == 'error':
                    cp1, cp2, (errors1, errors2) = coupled_non_negative_parafac(
                        tensor1=self.tensor1,
                        tensor2=self.tensor2,
                        rank=r,
                        non_shared_mode=self.non_shared_mode,
                        n_iter_max=n_iter_max,
                        tol=tol,
                        random_state=rs,
                        verbose=verbose,
                        balance_errors=self.balance_errors,
                        **kwargs
                    )

                    # Calculate combined error
                    N1 = self.tensor1.shape[self.non_shared_mode]
                    N2 = self.tensor2.shape[self.non_shared_mode]

                    if self.balance_errors:
                        total_N = N1 + N2
                        weight1 = total_N / N1
                        weight2 = total_N / N2
                        combined_error = (weight1 * errors1[-1] + weight2 * errors2[-1]) / (weight1 + weight2)
                    else:
                        combined_error = (errors1[-1] + errors2[-1]) / 2

                    run_errors.append(combined_error)

                elif metric == 'similarity':
                    # For similarity, we need to implement correlation index between runs
                    # This is more complex and would require storing factors from each run
                    # For now, we'll use the error metric as a placeholder
                    raise NotImplementedError("Similarity metric for coupled tensors not yet implemented")

            all_loss.append(run_errors)

        return np.array(all_loss).T

    def explained_variance(self):
        """Calculate explained variance for coupled factorization"""
        if self.tl_object1 is None or self.tl_object2 is None:
            raise ValueError("Must run compute_coupled_tensor_factorization first")

        # Calculate explained variance for each tensor
        rec_tensor1 = self.tl_object1.to_tensor()
        rec_tensor2 = self.tl_object2.to_tensor()

        # Apply masks if available
        tensor1 = self.tensor1
        tensor2 = self.tensor2

        if self.mask1 is not None:
            tensor1 = tensor1 * self.mask1
            rec_tensor1 = rec_tensor1 * self.mask1

        if self.mask2 is not None:
            tensor2 = tensor2 * self.mask2
            rec_tensor2 = rec_tensor2 * self.mask2

        # Calculate explained variance for each tensor
        def calc_explained_var(original, reconstructed):
            y_diff_avg = tl.mean(original - reconstructed)
            numerator = tl.norm(original - reconstructed - y_diff_avg)
            tensor_avg = tl.mean(original)
            denominator = tl.norm(original - tensor_avg)

            if denominator == 0.:
                return 0.0
            else:
                return 1. - (numerator / denominator)

        ev1 = calc_explained_var(tensor1, rec_tensor1).item()
        ev2 = calc_explained_var(tensor2, rec_tensor2).item()

        # Return weighted average based on tensor sizes
        N1 = np.prod(tensor1.shape)
        N2 = np.prod(tensor2.shape)
        total_elements = N1 + N2

        weighted_ev = (ev1 * N1 + ev2 * N2) / total_elements
        return weighted_ev

    def get_top_factor_elements(self, order_name, factor_name, top_number=10, tensor='unified'):
        """
        Get top elements for a given factor

        Parameters
        ----------
        order_name : str
            Name of the tensor dimension

        factor_name : str
            Name of the factor (e.g., 'Factor 1')

        top_number : int, default=10
            Number of top elements to return

        tensor : str, default='unified'
            Which factor set to use {'unified', 'tensor1', 'tensor2'}

        Returns
        -------
        top_elements : pandas.DataFrame
            Top elements for the specified factor
        """
        if tensor == 'unified':
            factors = self.factors
        elif tensor == 'tensor1':
            factors = self.factors1
        elif tensor == 'tensor2':
            factors = self.factors2
        else:
            raise ValueError("tensor must be 'unified', 'tensor1', or 'tensor2'")

        if order_name not in factors:
            raise ValueError(f"Order '{order_name}' not found in {tensor} factors")

        top_elements = factors[order_name][factor_name].sort_values(ascending=False).head(top_number)
        return top_elements

    def export_factor_loadings(self, filename, include_separate=True):
        """
        Export factor loadings to Excel file

        Parameters
        ----------
        filename : str
            Path for the output Excel file

        include_separate : bool, default=True
            Whether to include separate tensor1 and tensor2 factors
        """
        with pd.ExcelWriter(filename) as writer:
            # Export unified factors
            for k, v in self.factors.items():
                v.to_excel(writer, sheet_name=f'Unified_{k}')

            if include_separate:
                # Export tensor1 factors
                for k, v in self.factors1.items():
                    v.to_excel(writer, sheet_name=f'T1_{k}')

                # Export tensor2 factors
                for k, v in self.factors2.items():
                    v.to_excel(writer, sheet_name=f'T2_{k}')

        print(f'Coupled tensor factor loadings saved to {filename}')

    @property
    def shape(self):
        """Return shapes of both tensors"""
        return (self.tensor1.shape, self.tensor2.shape)

    def to_device(self, device):
        """Move tensors to specified device"""
        try:
            self.tensor1 = tl.tensor(self.tensor1, device=device)
            self.tensor2 = tl.tensor(self.tensor2, device=device)
            if self.mask1 is not None:
                self.mask1 = tl.tensor(self.mask1, device=device)
            if self.mask2 is not None:
                self.mask2 = tl.tensor(self.mask2, device=device)
        except:
            print('Device not available or backend does not support this device.')
            self.tensor1 = tl.tensor(self.tensor1)
            self.tensor2 = tl.tensor(self.tensor2)
            if self.mask1 is not None:
                self.mask1 = tl.tensor(self.mask1)
            if self.mask2 is not None:
                self.mask2 = tl.tensor(self.mask2)