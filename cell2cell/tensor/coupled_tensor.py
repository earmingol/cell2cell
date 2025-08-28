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
   _multiple_runs_coupled_elbow_analysis,
   _process_mode_mapping,
   _validate_tensors
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
        - dict: {'shared': [(t1_mode, t2_mode), ...], 'tensor1_only': [...], 'tensor2_only': [...]}
        - int/list: non_shared_modes (for backward compatibility with same-dimension tensors)

    tensor1_name : str, default='Tensor1'
        Name for the first tensor (used in factor labeling).

    tensor2_name : str, default='Tensor2'
        Name for the second tensor (used in factor labeling).

    balance_errors : bool, default=True
        Whether to balance the errors based on tensor-specific dimensions.

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

    def compute_tensor_factorization(self, rank, tf_type='coupled_non_negative_cp', init='svd',
                                     svd='truncated_svd', random_state=None, runs=1,
                                     normalize_loadings=True, var_ordered_factors=True,
                                     n_iter_max=100, tol=10e-7, verbose=False, **kwargs):
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
            if self.balance_errors:
                # Automatically derive tensor-specific modes
                shared_t1_modes = set([pair[0] for pair in self.mode_mapping.get('shared', [])])
                shared_t2_modes = set([pair[1] for pair in self.mode_mapping.get('shared', [])])
                tensor1_only = [i for i in range(tl.ndim(self.tensor1)) if i not in shared_t1_modes]
                tensor2_only = [i for i in range(tl.ndim(self.tensor2)) if i not in shared_t2_modes]

                nonshared_size1 = np.prod([self.tensor1.shape[i] for i in tensor1_only]) if tensor1_only else 1
                nonshared_size2 = np.prod([self.tensor2.shape[i] for i in tensor2_only]) if tensor2_only else 1
                total_nonshared = nonshared_size1 + nonshared_size2
                if total_nonshared > 0:
                    weight1 = total_nonshared / nonshared_size1 if nonshared_size1 > 0 else 1.0
                    weight2 = total_nonshared / nonshared_size2 if nonshared_size2 > 0 else 1.0
                    combined_error = (weight1 * error1 + weight2 * error2) / (weight1 + weight2)
                else:
                    combined_error = (error1 + error2) / 2
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

    def elbow_rank_selection(self, upper_rank=50, runs=20, tf_type='coupled_non_negative_cp',
                             init='random', svd='truncated_svd', metric='error', random_state=None,
                             n_iter_max=100, tol=10e-7, automatic_elbow=True, manual_elbow=None,
                             smooth=False, mask1=None, mask2=None, ci='std', figsize=(4, 2.25),
                             fontsize=14, filename=None, output_fig=True, verbose=False, **kwargs):
        '''
        Elbow analysis on the error achieved by the Coupled Tensor Factorization.
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
                balance_errors=self.balance_errors,
                **kwargs
            )
            loss = [(l[0], l[1].item() if hasattr(l[1], 'item') else l[1]) for l in loss]
            all_loss = np.array([[l[1] for l in loss]])
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