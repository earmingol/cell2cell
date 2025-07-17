# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorly as tl

from collections import OrderedDict
from tqdm import tqdm

from cell2cell.tensor.factorization import _compute_elbow, _compute_norm_error
from cell2cell.plotting.tensor_plot import plot_elbow, plot_multiple_run_elbow
from cell2cell.preprocessing.signal import smooth_curve
from cell2cell.tensor.coupled_factorization import (
    _compute_coupled_tensor_factorization,
    _run_coupled_elbow_analysis,
    _multiple_runs_coupled_elbow_analysis
)


class CoupledInteractionTensor():
    '''
    Coupled Tensor Factorization for two interaction tensors that share all but one dimension.

    This class performs simultaneous non-negative CP decomposition on two tensors that share
    all dimensions except one. The shared dimensions will have the same factor matrices,
    while the non-shared dimension will have separate factor matrices.

    Parameters
    ----------
    tensor1 : cell2cell.tensor.BaseTensor
        First interaction tensor (e.g., InteractionTensor, PreBuiltTensor).

    tensor2 : cell2cell.tensor.BaseTensor
        Second interaction tensor (e.g., InteractionTensor, PreBuiltTensor).

    non_shared_mode : int
        The mode (dimension) that differs between the two tensors.

    tensor1_name : str, default='Tensor1'
        Name for the first tensor (used in factor labeling).

    tensor2_name : str, default='Tensor2'
        Name for the second tensor (used in factor labeling).

    balance_errors : bool, default=True
        Whether to balance the errors based on the size of non-shared dimensions.

    device : str, default=None
        Device to use when backend allows using multiple devices.

    Attributes
    ----------
    tensor1 : tensorly.tensor
        First tensor object.

    tensor2 : tensorly.tensor
        Second tensor object.

    non_shared_mode : int
        The dimension that differs between tensors.

    cp1 : CPTensor
        CP decomposition result for tensor1.

    cp2 : CPTensor
        CP decomposition result for tensor2.

    factors1 : dict
        Factor loadings for tensor1.

    factors2 : dict
        Factor loadings for tensor2.

    factors : dict
        Combined factor loadings with shared and non-shared factors.

    order_names1 : list
        Element names for each dimension of tensor1.

    order_names2 : list
        Element names for each dimension of tensor2.

    order_labels1 : list
        Dimension labels for tensor1.

    order_labels2 : list
        Dimension labels for tensor2.
    '''

    def __init__(self, tensor1, tensor2, non_shared_mode, tensor1_name='Tensor1',
                 tensor2_name='Tensor2', balance_errors=True, device=None):
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

        # Store location information for zeros and NaNs
        if hasattr(tensor1, 'loc_nans') and tensor1.loc_nans is not None:
            self.loc_nans1 = tensor1.loc_nans
        else:
            self.loc_nans1 = None

        if hasattr(tensor2, 'loc_nans') and tensor2.loc_nans is not None:
            self.loc_nans2 = tensor2.loc_nans
        else:
            self.loc_nans2 = None

        if hasattr(tensor1, 'loc_zeros') and tensor1.loc_zeros is not None:
            self.loc_zeros1 = tensor1.loc_zeros
        else:
            self.loc_zeros1 = None

        if hasattr(tensor2, 'loc_zeros') and tensor2.loc_zeros is not None:
            self.loc_zeros2 = tensor2.loc_zeros
        else:
            self.loc_zeros2 = None

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

    def compute_tensor_factorization(self, rank, tf_type='coupled_non_negative_cp', init='svd',
                                     svd='truncated_svd', random_state=None, runs=1,
                                     normalize_loadings=True, var_ordered_factors=True,
                                     n_iter_max=100, tol=10e-7, verbose=False, **kwargs):
        '''
        Performs coupled tensor factorization on both tensors.
        There are no returns, instead the attributes factors and rank
        of the Tensor class are updated.

        Parameters
        ----------
        rank : int
            Number of components for the factorization.

        tf_type : str, default='coupled_non_negative_cp'
            Type of Tensor Factorization. Currently only supports 'coupled_non_negative_cp'.

        init : str, default='svd'
            Initialization method. Options are {'svd', 'random'}.

        svd : str, default='truncated_svd'
            SVD function to use.

        random_state : int, default=None
            Random state for reproducibility.

        runs : int, default=1
            Number of models to choose among and find the lowest error.
            This helps to avoid local minima when using runs > 1.

        normalize_loadings : boolean, default=True
            Whether normalizing the loadings in each factor to unit
            Euclidean length.

        var_ordered_factors : boolean, default=True
            Whether ordering factors by the variance they explain. The order is from
            highest to lowest variance. `normalize_loadings` must be True. Otherwise,
            this parameter is ignored.

        n_iter_max : int, default=100
            Maximum number of iterations.

        tol : float, default=1e-7
            Convergence tolerance.

        verbose : boolean, default=False
            Whether printing or not steps of the analysis.

        **kwargs : dict
            Extra arguments for the tensor factorization.
        '''
        best_error = np.inf
        best_cp1, best_cp2 = None, None

        if kwargs is None:
            kwargs = {'return_errors': True}
        else:
            kwargs['return_errors'] = True

        for run in tqdm(range(runs), disable=(runs == 1)):
            if random_state is not None:
                rs = random_state + run
            else:
                rs = None

            # Perform coupled factorization
            cp1, cp2, (errors1, errors2) = _compute_coupled_tensor_factorization(
                tensor1=self.tensor1,
                tensor2=self.tensor2,
                rank=rank,
                non_shared_mode=self.non_shared_mode,
                mask1=self.mask1,
                mask2=self.mask2,
                tf_type=tf_type,
                init=init,
                svd=svd,
                random_state=rs,
                n_iter_max=n_iter_max,
                tol=tol,
                verbose=verbose,
                balance_errors=self.balance_errors,
                **kwargs
            )

            # Calculate combined error for comparison
            if self.mask1 is None:
                error1 = tl.to_numpy(errors1[-1])
            else:
                error1 = _compute_norm_error(self.tensor1, cp1, self.mask1)

            if self.mask2 is None:
                error2 = tl.to_numpy(errors2[-1])
            else:
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

            if combined_error < best_error:
                best_error = combined_error
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

    def elbow_rank_selection(self, upper_rank=50, runs=20, tf_type='coupled_non_negative_cp',
                             init='random', svd='truncated_svd', metric='error', random_state=None,
                             n_iter_max=100, tol=10e-7, automatic_elbow=True, manual_elbow=None,
                             smooth=False, mask1=None, mask2=None, ci='std', figsize=(4, 2.25),
                             fontsize=14, filename=None, output_fig=True, verbose=False, **kwargs):
        '''
        Elbow analysis on the error achieved by the Coupled Tensor Factorization for
        selecting the number of factors to use. A plot is made with the results.

        Parameters
        ----------
        upper_rank : int, default=50
            Upper bound of ranks to explore with the elbow analysis.

        runs : int, default=20
            Number of tensor factorization performed for a given rank.

        tf_type : str, default='coupled_non_negative_cp'
            Type of Tensor Factorization (currently only supports 'coupled_non_negative_cp')

        init : str, default='random'
            Initialization method {'svd', 'random'}

        svd : str, default='truncated_svd'
            Function to compute the SVD, acceptable values in tensorly.SVD_FUNS

        metric : str, default='error'
            Metric for elbow analysis. Currently only supports 'error'.

        random_state : int, default=None
            Random seed

        n_iter_max : int, default=100
            Maximum iterations per factorization

        tol : float, default=10e-7
            Convergence tolerance

        automatic_elbow : boolean, default=True
            Whether using an automatic strategy to find the elbow.

        manual_elbow : int, default=None
            Manual elbow selection

        smooth : boolean, default=False
            Whether to smooth the curve

        mask1 : ndarray list, default=None
            Mask for the first tensor

        mask2 : ndarray list, default=None
            Mask for the second tensor

        ci : str, default='std'
            Confidence interval type

        figsize : tuple, default=(4, 2.25)
            Figure size

        fontsize : int, default=14
            Font size for labels

        filename : str, default=None
            Path to save figure

        output_fig : boolean, default=True
            Whether to generate figure

        verbose : boolean, default=False
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
        assert metric in ['error'], "`metric` must be 'error' for coupled factorization"
        ylabel = 'Normalized Error'

        if verbose:
            print('Running Coupled Elbow Analysis')

        # Use masks from tensors if not provided
        if mask1 is None:
            mask1 = self.mask1
        if mask2 is None:
            mask2 = self.mask2

        # Run analysis
        if runs == 1:
            loss = _run_coupled_elbow_analysis(
                tensor1=self.tensor1,
                tensor2=self.tensor2,
                non_shared_mode=self.non_shared_mode,
                upper_rank=upper_rank,
                tf_type=tf_type,
                init=init,
                svd=svd,
                random_state=random_state,
                mask1=mask1,
                mask2=mask2,
                n_iter_max=n_iter_max,
                tol=tol,
                verbose=verbose,
                balance_errors=self.balance_errors,
                **kwargs
            )
            loss = [(l[0], l[1].item() if hasattr(l[1], 'item') else l[1]) for l in loss]
            all_loss = np.array([[l[1] for l in loss]])
        else:
            all_loss = _multiple_runs_coupled_elbow_analysis(
                tensor1=self.tensor1,
                tensor2=self.tensor2,
                non_shared_mode=self.non_shared_mode,
                upper_rank=upper_rank,
                runs=runs,
                tf_type=tf_type,
                init=init,
                svd=svd,
                metric=metric,
                random_state=random_state,
                mask1=mask1,
                mask2=mask2,
                n_iter_max=n_iter_max,
                tol=tol,
                verbose=verbose,
                balance_errors=self.balance_errors,
                **kwargs
            )
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
                                 ylabel=ylabel, fontsize=fontsize, filename=filename)
            else:
                fig = plot_multiple_run_elbow(all_loss=all_loss, ci=ci, elbow=rank,
                                              figsize=figsize, ylabel=ylabel,
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

    def explained_variance(self):
        '''Calculate explained variance for coupled factorization'''
        if self.tl_object1 is None or self.tl_object2 is None:
            raise ValueError("Must run compute_tensor_factorization first")

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
        '''
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
        '''
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

    def export_factor_loadings(self, filename, save_separate=False):
        '''
        Export factor loadings to Excel file

        Parameters
        ----------
        filename : str
            Path for the output Excel file

        save_separate : bool, default=False
            Whether to save tensor1 and tensor2 factors separately
        '''
        writer = pd.ExcelWriter(filename)
        if save_separate:
            # Export tensor1 factors
            for k, v in self.factors1.items():
                v.to_excel(writer, sheet_name=f'T1_{k}')

            # Export tensor2 factors
            for k, v in self.factors2.items():
                v.to_excel(writer, sheet_name=f'T2_{k}')
        else:
            # Export unified factors
            for k, v in self.factors.items():
                v.to_excel(writer, sheet_name=f'{k}')

        writer.close()
        print(f'Coupled tensor factor loadings saved to {filename}')

    @property
    def shape(self):
        '''Return shapes of both tensors'''
        return (self.tensor1.shape, self.tensor2.shape)

    def to_device(self, device):
        '''Move tensors to specified device'''
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

    def copy(self):
        '''Performs a deep copy of this object'''
        import copy
        return copy.deepcopy(self)

    def write_file(self, filename):
        '''
        Exports this object into a pickle file

        Parameters
        ----------
        filename : str
            Complete path to the file wherein the variable will be stored
        '''
        from cell2cell.io.save_data import export_variable_with_pickle
        export_variable_with_pickle(self, filename=filename)

    def excluded_value_fraction(self, tensor='both'):
        '''
        Returns the fraction of excluded values in the tensor(s),
        given the values that are masked

        Parameters
        ----------
        tensor : str, default='both'
            Which tensor to analyze {'tensor1', 'tensor2', 'both', 'combined'}

        Returns
        -------
        excluded_fraction : float or dict
            Fraction of missing/excluded values. If tensor='both', returns dict
            with separate values. If tensor='combined', returns weighted average.
        '''
        if tensor == 'tensor1':
            if self.mask1 is None:
                return 0.0
            else:
                fraction = tl.sum(self.mask1) / tl.prod(tl.tensor(self.tensor1.shape))
                return 1.0 - fraction.item()

        elif tensor == 'tensor2':
            if self.mask2 is None:
                return 0.0
            else:
                fraction = tl.sum(self.mask2) / tl.prod(tl.tensor(self.tensor2.shape))
                return 1.0 - fraction.item()

        elif tensor == 'both':
            return {
                'tensor1': self.excluded_value_fraction('tensor1'),
                'tensor2': self.excluded_value_fraction('tensor2')
            }

        elif tensor == 'combined':
            # Weighted average based on tensor sizes
            exc1 = self.excluded_value_fraction('tensor1')
            exc2 = self.excluded_value_fraction('tensor2')
            size1 = np.prod(self.tensor1.shape)
            size2 = np.prod(self.tensor2.shape)
            total_size = size1 + size2
            return (exc1 * size1 + exc2 * size2) / total_size
        else:
            raise ValueError("tensor must be 'tensor1', 'tensor2', 'both', or 'combined'")

    def sparsity_fraction(self, tensor='both'):
        '''
        Returns the fraction of values that are zeros in the tensor(s),
        given the values that are in loc_zeros

        Parameters
        ----------
        tensor : str, default='both'
            Which tensor to analyze {'tensor1', 'tensor2', 'both', 'combined'}

        Returns
        -------
        sparsity_fraction : float or dict
            Fraction of values that are real zeros
        '''
        if tensor == 'tensor1':
            if self.loc_zeros1 is None:
                return 0.0
            else:
                sparsity = tl.sum(self.loc_zeros1) / tl.prod(tl.tensor(self.tensor1.shape))
                return sparsity.item()

        elif tensor == 'tensor2':
            if self.loc_zeros2 is None:
                return 0.0
            else:
                sparsity = tl.sum(self.loc_zeros2) / tl.prod(tl.tensor(self.tensor2.shape))
                return sparsity.item()

        elif tensor == 'both':
            return {
                'tensor1': self.sparsity_fraction('tensor1'),
                'tensor2': self.sparsity_fraction('tensor2')
            }

        elif tensor == 'combined':
            # Weighted average based on tensor sizes
            spar1 = self.sparsity_fraction('tensor1')
            spar2 = self.sparsity_fraction('tensor2')
            size1 = np.prod(self.tensor1.shape)
            size2 = np.prod(self.tensor2.shape)
            total_size = size1 + size2
            return (spar1 * size1 + spar2 * size2) / total_size
        else:
            raise ValueError("tensor must be 'tensor1', 'tensor2', 'both', or 'combined'")

    def missing_fraction(self, tensor='both'):
        '''
        Returns the fraction of values that are missing (NaNs) in the tensor(s),
        given the values that are in loc_nans

        Parameters
        ----------
        tensor : str, default='both'
            Which tensor to analyze {'tensor1', 'tensor2', 'both', 'combined'}

        Returns
        -------
        missing_fraction : float or dict
            Fraction of values that are missing (NaNs)
        '''
        if tensor == 'tensor1':
            if self.loc_nans1 is None:
                return 0.0
            else:
                missing = tl.sum(self.loc_nans1) / tl.prod(tl.tensor(self.tensor1.shape))
                return missing.item()

        elif tensor == 'tensor2':
            if self.loc_nans2 is None:
                return 0.0
            else:
                missing = tl.sum(self.loc_nans2) / tl.prod(tl.tensor(self.tensor2.shape))
                return missing.item()

        elif tensor == 'both':
            return {
                'tensor1': self.missing_fraction('tensor1'),
                'tensor2': self.missing_fraction('tensor2')
            }

        elif tensor == 'combined':
            # Weighted average based on tensor sizes
            miss1 = self.missing_fraction('tensor1')
            miss2 = self.missing_fraction('tensor2')
            size1 = np.prod(self.tensor1.shape)
            size2 = np.prod(self.tensor2.shape)
            total_size = size1 + size2
            return (miss1 * size1 + miss2 * size2) / total_size
        else:
            raise ValueError("tensor must be 'tensor1', 'tensor2', 'both', or 'combined'")