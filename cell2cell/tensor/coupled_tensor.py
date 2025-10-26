# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorly as tl

from collections import OrderedDict
from tqdm import tqdm

from cell2cell.tensor.factorization import _compute_elbow, _compute_norm_error
from cell2cell.plotting.tensor_plot import plot_coupled_elbow, plot_multiple_run_coupled_elbow, plot_coupled_factorization_errors
from cell2cell.preprocessing.signal import smooth_curve
from cell2cell.tensor.coupled_factorization import (
   _compute_coupled_tensor_factorization,
   _run_coupled_elbow_analysis,
   _multiple_runs_coupled_elbow_analysis,
   _process_mode_mapping,
   _validate_tensors,
   _compute_balancing_weights
)


class CoupledInteractionTensor():
    '''
    Coupled Tensor Factorization for two interaction tensors with flexible mode mapping.

    This class performs simultaneous non-negative CP decomposition on two tensors that can
    have different numbers of dimensions but share some modes. The mode mapping explicitly
    specifies which dimensions are shared and which are tensor-specific.

    Parameters
    ----------
    tensor1 : cell2cell.tensor.BaseTensor
        First interaction tensor (e.g., InteractionTensor, PreBuiltTensor).

    tensor2 : cell2cell.tensor.BaseTensor
        Second interaction tensor (e.g., InteractionTensor, PreBuiltTensor).

    mode_mapping : dict or int/list (for backward compatibility)
        Mode mapping specification. Can be:
        - dict: {'shared': [(t1_mode, t2_mode), ...]}     # Pairs of shared modes
        - int/list: non_shared_modes (for backward compatibility with same-dimension tensors)

    tensor1_name : str, default='Tensor1'
        Name for the first tensor (used in factor labeling).

    tensor2_name : str, default='Tensor2'
        Name for the second tensor (used in factor labeling).

    balance_errors : bool, default=True
        Whether to balance the errors based on tensor-specific dimensions.

    manual_weights : tuple, default=(0.5, 0.5)
            Manual weights (weight1, weight2) for importance of tensors in the factorization.
            Weights should be positive. Example: (2.0, 1.0) gives tensor1 twice
            the importance of tensor2 in both the factorization and the combined error metric.
            If None, automatic weight calculation is performed to have weigh1 and weight2
            inversely proportional to non-shared mode dimensions of each tensor.

    device : str, default=None
        Device to use when backend allows using multiple devices.

    Attributes
    ----------
    tensor1 : tensorly.tensor
        First tensor object.

    tensor2 : tensorly.tensor
        Second tensor object.

    mode_mapping : dict
        The mode mapping specification.

    cp1 : CPTensor
        CP decomposition result for tensor1.

    cp2 : CPTensor
        CP decomposition result for tensor2.

    factors1 : dict
        Factor loadings for tensor1.

    factors2 : dict
        Factor loadings for tensor2.

    factors : dict
        Combined factor loadings with shared and tensor-specific factors.

    factorization_errors1_ : list
        List of reconstruction errors for tensor1 at each iteration of the coupled
        tensor factorization. Only available after running compute_tensor_factorization.

    factorization_errors2_ : list
        List of reconstruction errors for tensor2 at each iteration of the coupled
        tensor factorization. Only available after running compute_tensor_factorization.

    combined_errors_ : list
        List of combined weighted reconstruction errors at each iteration of the coupled
        tensor factorization. The weighting follows the balance_errors parameter.
        Only available after running compute_tensor_factorization.
    '''

    def __init__(self, tensor1, tensor2, mode_mapping, tensor1_name='Tensor1',
                 tensor2_name='Tensor2', balance_errors=True, device=None):

        # Handle backward compatibility and validate inputs
        self.mode_mapping = _process_mode_mapping(tensor1.tensor, tensor2.tensor, mode_mapping)
        _validate_tensors(tensor1, tensor2, self.mode_mapping)

        # Store tensor objects and metadata
        self.tensor1 = tensor1.tensor
        self.tensor2 = tensor2.tensor
        self.tensor1_name = tensor1_name
        self.tensor2_name = tensor2_name
        self.balance_errors = balance_errors
        self.manual_weights = None

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

        # Initialize factorization error tracking
        self.factorization_errors1_ = None
        self.factorization_errors2_ = None
        self.combined_errors_ = None

    def compute_tensor_factorization(self, rank, tf_type='coupled_non_negative_cp', init='svd',
                                     svd='truncated_svd', random_state=None, runs=1,
                                     normalize_loadings=True, var_ordered_factors=True,
                                     n_iter_max=100, tol=10e-7, balance_errors=None, manual_weights=(0.5, 0.5),
                                     verbose=False, **kwargs):
        '''
        Performs coupled tensor factorization on both tensors.

        Parameters
        ----------
        rank : int
            Number of components for the factorization.

        tf_type : str, default='coupled_non_negative_cp'
            Type of Tensor Factorization.

        init : str, default='svd'
            Initialization method. Options are {'svd', 'random'}.

        svd : str, default='truncated_svd'
            SVD function to use.

        random_state : int, default=None
            Random state for reproducibility.

        runs : int, default=1
            Number of models to choose among and find the lowest error.

        normalize_loadings : boolean, default=True
            Whether normalizing the loadings in each factor.

        var_ordered_factors : boolean, default=True
            Whether ordering factors by variance explained.

        n_iter_max : int, default=100
            Maximum number of iterations.

        tol : float, default=1e-7
            Convergence tolerance.

        balance_errors : bool, default=None
            Whether to balance the errors based on tensor-specific dimensions.
            If None, valued used when initializing the CoupledTensor will be used.

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
        '''
        best_error = np.inf
        best_cp1, best_cp2 = None, None
        best_errors1, best_errors2 = None, None  # Track errors from best run
        best_weight1, best_weight2 = 1.0, 1.0  # Track weights from best run

        if balance_errors is None:
            balance_errors = self.balance_errors

        # Store manual weights if provided
        if manual_weights is not None:
            self.manual_weights = manual_weights

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
                mode_mapping=self.mode_mapping,
                mask1=self.mask1,
                mask2=self.mask2,
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

            # Calculate combined error for comparison
            if self.mask1 is None:
                error1 = tl.to_numpy(errors1[-1])
            else:
                error1 = _compute_norm_error(self.tensor1, cp1, self.mask1)

            if self.mask2 is None:
                error2 = tl.to_numpy(errors2[-1])
            else:
                error2 = _compute_norm_error(self.tensor2, cp2, self.mask2)

            # Calculate balancing weights (manual or automatic)
            weight1, weight2 = _compute_balancing_weights(
                self.tensor1, self.tensor2, self.mode_mapping,
                self.balance_errors, manual_weights
            )
            combined_error = (weight1 * error1 + weight2 * error2) / (weight1 + weight2)

            if combined_error < best_error:
                best_error = combined_error
                best_cp1, best_cp2 = cp1, cp2
                best_errors1, best_errors2 = errors1, errors2  # Store errors from best run
                best_weight1, best_weight2 = weight1, weight2  # Store weights

        if runs > 1:
            print(f'Best coupled model has a combined normalized error of: {best_error:.3f}')

        # Store results
        self.tl_object1 = best_cp1
        self.tl_object2 = best_cp2
        self.rank = rank

        # Store factorization errors from the best run
        if best_errors1 is not None and best_errors2 is not None:
            self.factorization_errors1_ = [tl.to_numpy(e) if hasattr(e, 'numpy') else e for e in best_errors1]
            self.factorization_errors2_ = [tl.to_numpy(e) if hasattr(e, 'numpy') else e for e in best_errors2]

            # Compute combined errors for each iteration
            self.combined_errors_ = []
            for e1, e2 in zip(best_errors1, best_errors2):
                e1_np = tl.to_numpy(e1) if hasattr(e1, 'numpy') else e1
                e2_np = tl.to_numpy(e2) if hasattr(e2, 'numpy') else e2

                if self.balance_errors:
                    combined = (best_weight1 * e1_np + best_weight2 * e2_np) / (best_weight1 + best_weight2)
                else:
                    combined = (e1_np + e2_np) / 2
                self.combined_errors_.append(combined)

        # Create factor DataFrames
        self._create_factor_dataframes(normalize_loadings, var_ordered_factors)

        # Calculate explained variance
        self.explained_variance_ = self.explained_variance()

    def _create_factor_dataframes(self, normalize_loadings, var_ordered_factors):
        """Create factor DataFrames and unified factors"""

        factor_names = ['Factor {}'.format(i) for i in range(1, self.rank + 1)]

        # Get order labels with defaults
        if self.order_labels1 is None:
            ndim1 = len(self.tensor1.shape)
            if ndim1 == 4:
                self.order_labels1 = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
            elif ndim1 > 4:
                self.order_labels1 = ['Contexts-{}'.format(i + 1) for i in range(ndim1 - 3)] + [
                    'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
            elif ndim1 == 3:
                self.order_labels1 = ['Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

        if self.order_labels2 is None:
            ndim2 = len(self.tensor2.shape)
            if ndim2 == 4:
                self.order_labels2 = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
            elif ndim2 > 4:
                self.order_labels2 = ['Contexts-{}'.format(i + 1) for i in range(ndim2 - 3)] + [
                    'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
            elif ndim2 == 3:
                self.order_labels2 = ['Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

        if normalize_loadings:
            self.norm_tl_object1 = tl.cp_tensor.cp_normalize(self.tl_object1)
            self.norm_tl_object2 = tl.cp_tensor.cp_normalize(self.tl_object2)
            (weights1, factors1) = self.norm_tl_object1
            (weights2, factors2) = self.norm_tl_object2
        else:
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

        # Create factor DataFrames for each tensor
        self.factors1 = OrderedDict(zip(self.order_labels1,
                                        [pd.DataFrame(f, index=idx, columns=factor_names)
                                         for f, idx in zip(factors1, self.order_names1)]))

        self.factors2 = OrderedDict(zip(self.order_labels2,
                                        [pd.DataFrame(f, index=idx, columns=factor_names)
                                         for f, idx in zip(factors2, self.order_names2)]))

        # Create unified factors based on mode mapping
        self.factors = OrderedDict()
        shared_pairs = self.mode_mapping.get('shared', [])

        # Automatically derive tensor-specific modes
        shared_t1_modes = set([pair[0] for pair in shared_pairs])
        shared_t2_modes = set([pair[1] for pair in shared_pairs])
        tensor1_only = [i for i in range(len(self.tensor1.shape)) if i not in shared_t1_modes]
        tensor2_only = [i for i in range(len(self.tensor2.shape)) if i not in shared_t2_modes]

        # Add shared factors (using tensor1's version)
        for t1_mode, t2_mode in shared_pairs:
            label = self.order_labels1[t1_mode]
            self.factors[label] = self.factors1[label]

        # Add tensor1-specific factors
        for mode in tensor1_only:
            label = self.order_labels1[mode]
            self.factors[label] = self.factors1[label]

        # Add tensor2-specific factors
        for mode in tensor2_only:
            label = self.order_labels2[mode]
            # Add suffix to distinguish if there's a name conflict
            if label in self.factors.keys():
                unified_label = f"{label}_{self.tensor2_name}"
            else:
                unified_label = label
            self.factors[unified_label] = self.factors2[label]

    def get_factorization_errors(self, plot=False, tensor1_name=None, tensor2_name=None,
                                 figsize=(10, 5), fontsize=12, show_individual=True,
                                 filename=None):
        '''Retrieves the factorization errors across iterations for coupled tensor factorization.'''
        if self.factorization_errors1_ is None or self.factorization_errors2_ is None:
            print("No factorization errors available. Please run compute_tensor_factorization first.")
            return None

        errors = {
            'tensor1': self.factorization_errors1_,
            'tensor2': self.factorization_errors2_,
            'combined': self.combined_errors_
        }

        if plot:
            t1_name = tensor1_name if tensor1_name is not None else self.tensor1_name
            t2_name = tensor2_name if tensor2_name is not None else self.tensor2_name

            fig = plot_coupled_factorization_errors(
                errors1=self.factorization_errors1_,
                errors2=self.factorization_errors2_,
                combined_errors=self.combined_errors_,
                tensor1_name=t1_name,
                tensor2_name=t2_name,
                figsize=figsize,
                fontsize=fontsize,
                show_individual=show_individual,
                filename=filename
            )
            return errors, fig
        else:
            return errors

    def elbow_rank_selection(self, upper_rank=50, runs=20, tf_type='coupled_non_negative_cp',
                             init='random', svd='truncated_svd', metric='error', random_state=None,
                             n_iter_max=100, tol=10e-7, automatic_elbow=True, manual_elbow=None,
                             smooth=False, mask1=None, mask2=None, balance_errors=None, manual_weights=(0.5, 0.5),
                             ci='std', figsize=(4, 2.25), fontsize=14, filename=None,
                             output_fig=True, show_individual=False, verbose=False, **kwargs):
        '''
        Elbow analysis on the error/similarity achieved by the Coupled Tensor Factorization.

        Parameters
        ----------
        upper_rank : int, default=50
            Upper bound of ranks to explore with the elbow analysis.

        runs : int, default=20
            Number of tensor factorization performed for a given rank.

        tf_type : str, default='coupled_non_negative_cp'
            Type of Tensor Factorization.

        init : str, default='random'
            Initialization method. {'svd', 'random'}

        svd : str, default='truncated_svd'
            Function to compute the SVD.

        metric : str, default='error'
            Metric to perform the elbow analysis.

            - 'error' : Normalized error to compute the elbow.
            - 'similarity' : Similarity based on CorrIndex (1-CorrIndex).

        random_state : int, default=None
            Seed for randomization.

        n_iter_max : int, default=100
            Maximum number of iterations.

        tol : float, default=1e-7
            Convergence tolerance.

        automatic_elbow : boolean, default=True
            Whether using an automatic strategy to find the elbow.

        manual_elbow : int, default=None
            Rank to highlight. Considered only when `automatic_elbow=False`.

        smooth : boolean, default=False
            Whether smoothing the curve.

        mask1 : tensorly.tensor, default=None
            Mask for the first tensor.

        mask2 : tensorly.tensor, default=None
            Mask for the second tensor.

        balance_errors : bool, default=None
            Whether to balance the errors based on tensor-specific dimensions.
            If None, valued used when initializing the CoupledTensor will be used.

        manual_weights : tuple, default=(0.5, 0.5)
            Manual weights (weight1, weight2) for importance of tensors in the factorization.
            Weights should be positive. Example: (2.0, 1.0) gives tensor1 twice
            the importance of tensor2 in both the factorization and the combined error metric.
            If None, automatic weight calculation is performed to have weigh1 and weight2
            inversely proportional to non-shared mode dimensions of each tensor.

        ci : str, default='std'
            Confidence interval. {'std', '95%'}

        figsize : tuple, default=(4, 2.25)
            Figure size.

        fontsize : int, default=14
            Font size for axis labels.

        filename : str, default=None
            Path to save the figure.

        output_fig : boolean, default=True
            Whether generating the figure.

        show_individual : boolean, default=False
            Whether to show individual tensor metrics alongside the combined metric.
            Applies to both 'error' and 'similarity' metrics when runs > 1.
            If True, plots will show tensor1, tensor2, and combined metrics.
            If False, only shows the combined metric.

        verbose : boolean, default=False
            Whether printing or not steps of the analysis.

        **kwargs : dict
            Extra arguments for the tensor factorization.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object made with matplotlib

        loss : dict
            Dictionary with 'tensor1', 'tensor2', and 'combined' keys, each containing
            a list of (rank, value) tuples for the respective metric.
        '''
        assert metric in ['similarity', 'error'], "`metric` must be either 'similarity' or 'error'"
        ylabel = {'similarity': 'Similarity\n(1-CorrIndex)', 'error': 'Normalized Error'}

        if verbose:
            print('Running Coupled Elbow Analysis')

        # Use masks from tensors if not provided
        if mask1 is None:
            mask1 = self.mask1
        if mask2 is None:
            mask2 = self.mask2

        if metric == 'similarity':
            assert runs > 1, "`runs` must be greater than 1 when `metric` = 'similarity'"

        if balance_errors is None:
            balance_errors = self.balance_errors

        # Run analysis
        if runs == 1:
            loss_dict = _run_coupled_elbow_analysis(
                tensor1=self.tensor1,
                tensor2=self.tensor2,
                mode_mapping=self.mode_mapping,
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
                balance_errors=balance_errors,
                manual_weights=manual_weights,
                **kwargs
            )
            # Convert to numeric for all metrics
            loss_dict = {
                'tensor1': [(l[0], l[1].item() if hasattr(l[1], 'item') else l[1]) for l in loss_dict['tensor1']],
                'tensor2': [(l[0], l[1].item() if hasattr(l[1], 'item') else l[1]) for l in loss_dict['tensor2']],
                'combined': [(l[0], l[1].item() if hasattr(l[1], 'item') else l[1]) for l in loss_dict['combined']]
            }
            # Create array structure for consistency with multiple runs
            all_loss = {
                'tensor1': np.array([[l[1] for l in loss_dict['tensor1']]]),
                'tensor2': np.array([[l[1] for l in loss_dict['tensor2']]]),
                'combined': np.array([[l[1] for l in loss_dict['combined']]])
            }
            loss = loss_dict
        else:
            all_loss = _multiple_runs_coupled_elbow_analysis(
                tensor1=self.tensor1,
                tensor2=self.tensor2,
                mode_mapping=self.mode_mapping,
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
                balance_errors=balance_errors,
                manual_weights=manual_weights,
                **kwargs
            )

            # all_loss is always a dict with 'tensor1', 'tensor2', 'combined'
            loss = {
                'tensor1': np.nanmean(all_loss['tensor1'], axis=0).tolist(),
                'tensor2': np.nanmean(all_loss['tensor2'], axis=0).tolist(),
                'combined': np.nanmean(all_loss['combined'], axis=0).tolist()
            }

            if smooth:
                loss['tensor1'] = smooth_curve(loss['tensor1'])
                loss['tensor2'] = smooth_curve(loss['tensor2'])
                loss['combined'] = smooth_curve(loss['combined'])

            # Convert to (rank, value) tuples
            loss_combined = [(i + 1, l) for i, l in enumerate(loss['combined'])]
            loss_t1 = [(i + 1, l) for i, l in enumerate(loss['tensor1'])]
            loss_t2 = [(i + 1, l) for i, l in enumerate(loss['tensor2'])]
            loss = {
                'tensor1': loss_t1,
                'tensor2': loss_t2,
                'combined': loss_combined
            }

        # Find elbow (always on combined metric)
        if automatic_elbow:
            rank = int(_compute_elbow(loss['combined']))
        else:
            rank = manual_elbow

        # Generate plot
        if output_fig:
            if runs == 1:
                # For runs=1, use the new coupled plotting function
                fig = plot_coupled_elbow(
                    loss_dict=loss, elbow=rank, figsize=figsize,
                    ylabel=ylabel[metric], fontsize=fontsize, filename=filename,
                    show_individual=show_individual,
                    tensor1_name=self.tensor1_name,
                    tensor2_name=self.tensor2_name
                )
            else:
                # For runs>1, use the existing coupled plotting function
                fig = plot_multiple_run_coupled_elbow(
                    all_loss=all_loss, ci=ci, elbow=rank,
                    figsize=figsize, ylabel=ylabel[metric],
                    smooth=smooth, fontsize=fontsize, filename=filename,
                    show_individual=show_individual,
                    tensor1_name=self.tensor1_name,
                    tensor2_name=self.tensor2_name
                )
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
        '''Get top elements for a given factor'''
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
        '''Export factor loadings to Excel file'''
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
        '''Exports this object into a pickle file'''
        from cell2cell.io.save_data import export_variable_with_pickle
        export_variable_with_pickle(self, filename=filename)

    def excluded_value_fraction(self, tensor='both'):
        '''Returns the fraction of excluded values in the tensor(s)'''
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
        '''Returns the fraction of values that are zeros in the tensor(s)'''
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
        '''Returns the fraction of values that are missing (NaNs) in the tensor(s)'''
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

    def reorder_metadata(self, metadata1, metadata2):
        '''
        Reorder metadata to match the factor ordering used by the coupled tensor.

        Parameters
        ----------
        metadata1 : list
            List of DataFrames/metadata for tensor1 factors, in original tensor1 mode order

        metadata2 : list
            List of DataFrames/metadata for tensor2 factors, in original tensor2 mode order

        Returns
        -------
        reordered_metadata : list
            Metadata reordered to match self.factors ordering:
            [shared_modes, tensor1_specific_modes, tensor2_specific_modes]
        '''
        if self.factors is None:
            raise ValueError("Must run compute_tensor_factorization first to determine factor ordering")

        # Get mode mappings
        shared_pairs = self.mode_mapping.get('shared', [])
        shared_t1_modes = set([pair[0] for pair in shared_pairs])
        shared_t2_modes = set([pair[1] for pair in shared_pairs])
        tensor1_only = [i for i in range(len(self.tensor1.shape)) if i not in shared_t1_modes]
        tensor2_only = [i for i in range(len(self.tensor2.shape)) if i not in shared_t2_modes]

        # Validate metadata lengths
        if len(metadata1) != len(self.tensor1.shape):
            raise ValueError(f"metadata1 must have {len(self.tensor1.shape)} elements, got {len(metadata1)}")
        if len(metadata2) != len(self.tensor2.shape):
            raise ValueError(f"metadata2 must have {len(self.tensor2.shape)} elements, got {len(metadata2)}")

        reordered_metadata = []

        # Add shared mode metadata (use tensor1's version)
        for t1_mode, t2_mode in shared_pairs:
            reordered_metadata.append(metadata1[t1_mode])

        # Add tensor1-specific mode metadata
        for mode in tensor1_only:
            reordered_metadata.append(metadata1[mode])

        # Add tensor2-specific mode metadata
        for mode in tensor2_only:
            reordered_metadata.append(metadata2[mode])

        return reordered_metadata