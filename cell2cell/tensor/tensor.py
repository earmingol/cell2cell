# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorly as tl

from collections import OrderedDict
from tqdm.auto import tqdm

from cell2cell.core.communication_scores import compute_ccc_matrix, aggregate_ccc_matrices
from cell2cell.preprocessing.find_elements import get_element_abundances, get_elements_over_fraction
from cell2cell.preprocessing.ppi import get_genes_from_complexes
from cell2cell.preprocessing.rnaseq import add_complexes_to_expression
from cell2cell.preprocessing.ppi import filter_ppi_by_proteins
from cell2cell.preprocessing.signal import smooth_curve
from cell2cell.tensor.factorization import _compute_tensor_factorization, _run_elbow_analysis, _multiple_runs_elbow_analysis, _compute_elbow, _compute_norm_error
from cell2cell.plotting.tensor_plot import plot_elbow, plot_multiple_run_elbow


class BaseTensor():
    '''Empty base tensor class that contains the main functions for the Tensor
    Factorization of a Communication Tensor

    Attributes
    ----------
    communication_score : str
        Type of communication score to infer the potential use of a given ligand-
        receptor pair by a pair of cells/tissues/samples.
        Available communication_scores are:

        - 'expression_mean' : Computes the average between the expression of a ligand
                              from a sender cell and the expression of a receptor on a
                              receiver cell.
        - 'expression_product' : Computes the product between the expression of a
                                ligand from a sender cell and the expression of a
                                receptor on a receiver cell.
        - 'expression_gmean' : Computes the geometric mean between the expression
                                   of a ligand from a sender cell and the
                                   expression of a receptor on a receiver cell.

    how : str
        Approach to consider cell types and genes present across multiple contexts.

        - 'inner' : Considers only cell types and genes that are present in all
                    contexts (intersection).
        - 'outer' : Considers all cell types and genes that are present
                    across contexts (union).
        - 'outer_genes' : Considers only cell types that are present in all
                          contexts (intersection), while all genes that are
                          present across contexts (union).
        - 'outer_cells' : Considers only genes that are present in all
                          contexts (intersection), while all cell types that are
                          present across contexts (union).

    outer_fraction : float
        Threshold to filter the elements when `how` includes any outer option.
        Elements with a fraction abundance across samples (in `rnaseq_matrices`)
        at least this threshold will be included. When this value is 0, considers
        all elements across the samples. When this value is 1, it acts as using
        `how='inner'`.

    tensor : tensorly.tensor
        Tensor object created with the library tensorly.

    genes : list
        List of strings detailing the genes used through all contexts. Obtained
        depending on the attribute 'how'.

    cells : list
        List of strings detailing the cells used through all contexts. Obtained
        depending on the attribute 'how'.

    order_names : list
        List of lists containing the string names of each element in each of the
        dimensions or orders in the tensor. For a 4D-Communication tensor, the first
        list should contain the names of the contexts, the second the names of the
        ligand-receptor interactions, the third the names of the sender cells and the
        fourth the names of the receiver cells.

    order_labels : list
        List of labels for dimensions or orders in the tensor.

    tl_object : ndarray list
        A tensorly object containing a list of initialized factors of the tensor
        decomposition where element `i` is of shape (tensor.shape[i], rank).

    norm_tl_object : ndarray list
        A tensorly object containing a list of initialized factors of the tensor
        decomposition where element `i` is of shape (tensor.shape[i], rank). This
        results from normalizing the factor loadings of the tl_object.

    factors : dict
        Ordered dictionary containing a dataframe with the factor loadings for each
        dimension/order of the tensor.

    rank : int
        Rank of the Tensor Factorization (number of factors to deconvolve the original
        tensor).

    mask : ndarray list
        Helps avoiding missing values during a tensor factorization. A mask should be
        a boolean array of the same shape as the original tensor and should be 0
        where the values are missing and 1 everywhere else.

    explained_variance : float
        Explained variance score for a tnesor factorization.

    explained_variance_ratio_ : ndarray list
        Percentage of variance explained by each of the factors. Only present when
        "normalize_loadings" is True. Otherwise, it is None.

    loc_nans : ndarray list
        An array of shape equal to `tensor` with ones where NaN values were assigned
        when building the tensor. Other values are zeros. It stores the
        location of the NaN values.

    loc_zeros : ndarray list
        An array of shape equal to `tensor` with ones where zeros that are not in
        `loc_nans` are located. Other values are assigned a zero. It tracks the
        real zero values rather than NaN values that were converted to zero.

    elbow_metric : str
        Stores the metric used to perform the elbow analysis (y-axis).

            - 'error' : Normalized error to compute the elbow.
            - 'similarity' : Similarity based on CorrIndex (1-CorrIndex).

    elbow_metric_mean : ndarray list
        Metric computed from the elbow analysis for each of the different rank
        evaluated. This list contains (X,Y) pairs where X values are the
        different ranks and Y values are the mean value of the metric employed.
        This mean is computed from multiple runs, or the values for just one run.
        Metric could be the normalized error of the decomposition or the similarity
        between multiple runs with different initialization, based on the
        CorrIndex.

    elbow_metric_raw : ndarray list
        Similar to `elbow_metric_mean`, but instead of containing (X, Y) pairs,
        it is an array of shape runs by ranks that were used for the analysis.
        It contains all the metrics for each run in each of the evaluated ranks.
    '''
    def __init__(self):
        # Save variables for this class
        self.communication_score = None
        self.how = None
        self.outer_fraction = None
        self.tensor = None
        self.genes = None
        self.cells = None
        self.order_names = [None, None, None, None]
        self.order_labels = None
        self.tl_object = None
        self.norm_tl_object = None
        self.factors = None
        self.rank = None
        self.mask = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.loc_nans = None
        self.loc_zeros = None
        self.elbow_metric = None
        self.elbow_metric_mean = None
        self.elbow_metric_raw = None

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def compute_tensor_factorization(self, rank, tf_type='non_negative_cp', init='svd', svd='numpy_svd', random_state=None,
                                     runs=1, normalize_loadings=True, var_ordered_factors=True, n_iter_max=100, tol=10e-7,
                                     verbose=False, **kwargs):
        '''Performs a Tensor Factorization.
        There are no returns, instead the attributes factors and rank
         of the Tensor class are updated.

        Parameters
        ----------
        rank : int
            Rank of the Tensor Factorization (number of factors to deconvolve the original
            tensor).

        tf_type : str, default='non_negative_cp'
            Type of Tensor Factorization.

            - 'non_negative_cp' : Non-negative PARAFAC through the traditional ALS.
            - 'non_negative_cp_hals' : Non-negative PARAFAC through the Hierarchical ALS.
                                       It reaches an optimal solution faster than the
                                       traditional ALS, but it does not allow a mask.
            - 'parafac' : PARAFAC through the traditional ALS. It allows negative loadings.
            - 'constrained_parafac' : PARAFAC through the traditional ALS. It allows
                                      negative loadings. Also, it incorporates L1 and L2
                                      regularization, includes a 'non_negative' option, and
                                      allows constraining the sparsity of the decomposition.
                                      For more information, see
                                      http://tensorly.org/stable/modules/generated/tensorly.decomposition.constrained_parafac.html#tensorly.decomposition.constrained_parafac


        init : str, default='svd'
            Initialization method for computing the Tensor Factorization.
            {‘svd’, ‘random’}

        svd : str, default='numpy_svd'
            Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS

        random_state : int, default=None
            Seed for randomization.

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

        tol : float, default=10e-7
            Tolerance for the decomposition algorithm to stop when the variation in
            the reconstruction error is less than the tolerance. Lower `tol` helps
            to improve the solution obtained from the decomposition, but it takes
            longer to run.

        n_iter_max : int, default=100
            Maximum number of iteration to reach an optimal solution with the
            decomposition algorithm. Higher `n_iter_max`helps to improve the solution
            obtained from the decomposition, but it takes longer to run.

        verbose : boolean, default=False
            Whether printing or not steps of the analysis.

        **kwargs : dict
            Extra arguments for the tensor factorization according to inputs in tensorly.
        '''
        tensor_dim = len(self.tensor.shape)
        best_err = np.inf
        tf = None

        if kwargs is None:
            kwargs = {'return_errors' : True}
        else:
            kwargs['return_errors'] = True

        for run in tqdm(range(runs), disable=(runs==1)):
            if random_state is not None:
                random_state_ = random_state + run
            else:
                random_state_ = None
            local_tf, errors = _compute_tensor_factorization(tensor=self.tensor,
                                                             rank=rank,
                                                             tf_type=tf_type,
                                                             init=init,
                                                             svd=svd,
                                                             random_state=random_state_,
                                                             mask=self.mask,
                                                             n_iter_max=n_iter_max,
                                                             tol=tol,
                                                             verbose=verbose,
                                                             **kwargs)
            # This helps to obtain proper error when the mask is not None.
            if self.mask is None:
                err = tl.to_numpy(errors[-1])
                if best_err > err:
                    best_err = err
                    tf = local_tf
            else:
                err = _compute_norm_error(self.tensor, local_tf, self.mask)
                if best_err > err:
                    best_err = err
                    tf = local_tf

        if runs > 1:
            print('Best model has a normalized error of: {0:.3f}'.format(best_err))

        self.tl_object = tf
        if normalize_loadings:
            self.norm_tl_object = tl.cp_tensor.cp_normalize(self.tl_object)

        factor_names = ['Factor {}'.format(i) for i in range(1, rank+1)]
        if self.order_labels is None:
            if tensor_dim == 4:
                order_labels = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
            elif tensor_dim > 4:
                order_labels = ['Contexts-{}'.format(i+1) for i in range(tensor_dim-3)] + ['Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
            elif tensor_dim == 3:
                order_labels = ['Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
            else:
                raise ValueError('Too few dimensions in the tensor')
        else:
            assert len(self.order_labels) == tensor_dim, "The length of order_labels must match the number of orders/dimensions in the tensor"
            order_labels = self.order_labels

        if normalize_loadings:
            (weights, factors) = self.norm_tl_object
            weights = tl.to_numpy(weights)
            if var_ordered_factors:
                w_order = weights.argsort()[::-1]
                factors = [tl.to_numpy(f)[:, w_order] for f in factors]
                self.explained_variance_ratio_ = weights[w_order] / sum(weights)
            else:
                factors = [tl.to_numpy(f) for f in factors]
                self.explained_variance_ratio_ = weights / sum(weights)

        else:
            (weights, factors) = self.tl_object
            self.explained_variance_ratio_ = None

        self.explained_variance_ = self.explained_variance()

        self.factors = OrderedDict(zip(order_labels,
                                       [pd.DataFrame(tl.to_numpy(f), index=idx, columns=factor_names) for f, idx in zip(factors, self.order_names)]))
        self.rank = rank

    def elbow_rank_selection(self, upper_rank=50, runs=20, tf_type='non_negative_cp', init='random', svd='numpy_svd',
                             metric='error', random_state=None, n_iter_max=100, tol=10e-7, automatic_elbow=True,
                             manual_elbow=None, smooth=False, mask=None, ci='std', figsize=(4, 2.25), fontsize=14,
                             filename=None, output_fig=True, verbose=False, **kwargs):
        '''Elbow analysis on the error achieved by the Tensor Factorization for
        selecting the number of factors to use. A plot is made with the results.

        Parameters
        ----------
        upper_rank : int, default=50
            Upper bound of ranks to explore with the elbow analysis.

        runs : int, default=20
            Number of tensor factorization performed for a given rank. Each
            factorization varies in the seed of initialization.

        tf_type : str, default='non_negative_cp'
            Type of Tensor Factorization.

            - 'non_negative_cp' : Non-negative PARAFAC through the traditional ALS.
            - 'non_negative_cp_hals' : Non-negative PARAFAC through the Hierarchical ALS.
                                       It reaches an optimal solution faster than the
                                       traditional ALS, but it does not allow a mask.
            - 'parafac' : PARAFAC through the traditional ALS. It allows negative loadings.
            - 'constrained_parafac' : PARAFAC through the traditional ALS. It allows
                                      negative loadings. Also, it incorporates L1 and L2
                                      regularization, includes a 'non_negative' option, and
                                      allows constraining the sparsity of the decomposition.
                                      For more information, see
                                      http://tensorly.org/stable/modules/generated/tensorly.decomposition.constrained_parafac.html#tensorly.decomposition.constrained_parafac

        init : str, default='svd'
            Initialization method for computing the Tensor Factorization.
            {‘svd’, ‘random’}

        svd : str, default='numpy_svd'
            Function to compute the SVD, acceptable values in tensorly.SVD_FUNS

        metric : str, default='error'
            Metric to perform the elbow analysis (y-axis)

            - 'error' : Normalized error to compute the elbow.
            - 'similarity' : Similarity based on CorrIndex (1-CorrIndex).

        random_state : int, default=None
            Seed for randomization.

        tol : float, default=10e-7
            Tolerance for the decomposition algorithm to stop when the variation in
            the reconstruction error is less than the tolerance. Lower `tol` helps
            to improve the solution obtained from the decomposition, but it takes
            longer to run.

        n_iter_max : int, default=100
            Maximum number of iteration to reach an optimal solution with the
            decomposition algorithm. Higher `n_iter_max`helps to improve the solution
            obtained from the decomposition, but it takes longer to run.

        automatic_elbow : boolean, default=True
            Whether using an automatic strategy to find the elbow. If True, the method
            implemented by the package kneed is used.

        manual_elbow : int, default=None
            Rank or number of factors to highlight in the curve of error achieved by
            the Tensor Factorization. This input is considered only when
            `automatic_elbow=True`

        smooth : boolean, default=False
            Whether smoothing the curve with a Savitzky-Golay filter.

        mask : ndarray list, default=None
            Helps avoiding missing values during a tensor factorization. A mask should be
            a boolean array of the same shape as the original tensor and should be 0
            where the values are missing and 1 everywhere else.

        ci : str, default='std'
            Confidence interval for representing the multiple runs in each rank.
            {'std', '95%'}

        figsize : tuple, default=(4, 2.25)
            Figure size, width by height

        fontsize : int, default=14
            Fontsize for axis labels.

        filename : str, default=None
            Path to save the figure of the elbow analysis. If None, the figure is not
            saved.

        output_fig : boolean, default=True
            Whether generating the figure with matplotlib.

        verbose : boolean, default=False
            Whether printing or not steps of the analysis.

        **kwargs : dict
            Extra arguments for the tensor factorization according to inputs in
            tensorly.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object made with matplotlib

        loss : list
            List of normalized errors for each rank. Here the errors are te average
            across distinct runs for each rank.
        '''
        assert metric in ['similarity', 'error'], "`metric` must be either 'similarity' or 'error'"
        ylabel = {'similarity' : 'Similarity\n(1-CorrIndex)', 'error' : 'Normalized Error'}

        # Run analysis
        if verbose:
            print('Running Elbow Analysis')

        if mask is None:
            if self.mask is not None:
                mask = self.mask

        if metric == 'similarity':
            assert runs > 1, "`runs` must be greater than 1 when `metric` = 'similarity'"
        if runs == 1:
            loss = _run_elbow_analysis(tensor=self.tensor,
                                       upper_rank=upper_rank,
                                       tf_type=tf_type,
                                       init=init,
                                       svd=svd,
                                       random_state=random_state,
                                       mask=mask,
                                       n_iter_max=n_iter_max,
                                       tol=tol,
                                       verbose=verbose,
                                       **kwargs
                                       )
            loss = [(l[0], l[1].item()) for l in loss]
            all_loss = np.array([[l[1] for l in loss]])
            if automatic_elbow:
                if smooth:
                    loss_ = [l[1] for l in loss]
                    loss = smooth_curve(loss_)
                    loss = [(i + 1, l) for i, l in enumerate(loss)]
                rank = int(_compute_elbow(loss))
            else:
                rank = manual_elbow
            if output_fig:
                fig = plot_elbow(loss=loss,
                                 elbow=rank,
                                 figsize=figsize,
                                 ylabel=ylabel[metric],
                                 fontsize=fontsize,
                                 filename=filename)
            else:
                fig = None
        elif runs > 1:
            all_loss = _multiple_runs_elbow_analysis(tensor=self.tensor,
                                                     upper_rank=upper_rank,
                                                     runs=runs,
                                                     tf_type=tf_type,
                                                     init=init,
                                                     svd=svd,
                                                     metric=metric,
                                                     random_state=random_state,
                                                     mask=mask,
                                                     n_iter_max=n_iter_max,
                                                     tol=tol,
                                                     verbose=verbose,
                                                     **kwargs
                                                     )

            # Same outputs as runs = 1
            loss = np.nanmean(all_loss, axis=0).tolist()
            if smooth:
                loss = smooth_curve(loss)
            loss = [(i + 1, l) for i, l in enumerate(loss)]

            if automatic_elbow:
                rank = int(_compute_elbow(loss))
            else:
                rank = manual_elbow

            if output_fig:
                fig = plot_multiple_run_elbow(all_loss=all_loss,
                                              ci=ci,
                                              elbow=rank,
                                              figsize=figsize,
                                              ylabel=ylabel[metric],
                                              smooth=smooth,
                                              fontsize=fontsize,
                                              filename=filename)
            else:
                fig = None

        else:
            assert runs > 0, "Input runs must be an integer greater than 0"

        # Store results
        self.rank = rank
        self.elbow_metric = metric
        self.elbow_metric_mean = loss
        self.elbow_metric_raw = all_loss

        if self.rank is not None:
            assert(isinstance(rank, int)), 'rank must be an integer.'
            print('The rank at the elbow is: {}'.format(self.rank))
        return fig, loss

    def get_top_factor_elements(self, order_name, factor_name, top_number=10):
        '''Obtains the top-elements with higher loadings for a given factor

        Parameters
        ----------
        order_name : str
            Name of the dimension/order in the tensor according to the keys of the
            dictionary in BaseTensor.factors. The attribute factors is built once the
            tensor factorization is run.

        factor_name : str
            Name of one of the factors. E.g., 'Factor 1'

        top_number : int, default=10
            Number of top-elements to return

        Returns
        -------
        top_elements : pandas.DataFrame
            A dataframe with the loadings of the top-elements for the given factor.
        '''
        top_elements = self.factors[order_name][factor_name].sort_values(ascending=False).head(top_number)
        return top_elements

    def export_factor_loadings(self, filename):
        '''Exports the factor loadings of the tensor into an Excel file

        Parameters
        ----------
        filename : str
            Full path and filename to store the file. E.g., '/home/user/Loadings.xlsx'
        '''
        writer = pd.ExcelWriter(filename)
        for k, v in self.factors.items():
            v.to_excel(writer, sheet_name=k)
        writer.save()
        print('Loadings of the tensor factorization were successfully saved into {}'.format(filename))

    def excluded_value_fraction(self):
        '''Returns the fraction of excluded values in the tensor,
        given the values that are masked in tensor.mask

        Returns
        -------
        excluded_fraction : float
            Fraction of missing/excluded values in the tensor.
        '''
        if self.mask is None:
            print("The interaction tensor does not have masked values")
            return 0.0
        else:
            fraction = tl.sum(self.mask) / tl.prod(tl.tensor(self.tensor.shape))
            excluded_fraction = 1.0 - fraction.item()
            return excluded_fraction

    def sparsity_fraction(self):
        '''Returns the fraction of values that are zeros in the tensor,
        given the values that are in tensor.loc_zeros

        Returns
        -------
        sparsity_fraction : float
            Fraction of values that are real zeros.
        '''
        if self.loc_zeros is None:
            print("The interaction tensor does not have zeros")
            return 0.0
        else:
            sparsity_fraction = tl.sum(self.loc_zeros) / tl.prod(tl.tensor(self.tensor.shape))
        sparsity_fraction = sparsity_fraction.item()
        return sparsity_fraction

    def missing_fraction(self):
        '''Returns the fraction of values that are missing (NaNs) in the tensor,
        given the values that are in tensor.loc_nans

        Returns
        -------
        missing_fraction : float
            Fraction of values that are real zeros.
        '''
        if self.loc_nans is None:
            print("The interaction tensor does not have missing values")
            return 0.0
        else:
            missing_fraction = tl.sum(self.loc_nans) / tl.prod(tl.tensor(self.tensor.shape))
        missing_fraction = missing_fraction.item()
        return missing_fraction

    def explained_variance(self):
        '''Computes the explained variance score for a tensor decomposition. Inspired on the
        function in sklearn.metrics.explained_variance_score.

        Returns
        -------
        explained_variance : float
            Explained variance score for a tnesor factorization.
        '''
        assert self.tl_object is not None, "Must run compute_tensor_factorization before using this method."
        tensor = self.tensor
        rec_tensor = self.tl_object.to_tensor()
        mask = self.mask

        if mask is not None:
            tensor = tensor * mask
            rec_tensor = tensor * mask

        y_diff_avg = tl.mean(tensor - rec_tensor)
        numerator = tl.norm(tensor - rec_tensor - y_diff_avg)

        tensor_avg = tl.mean(tensor)
        denominator = tl.norm(tensor - tensor_avg)

        if denominator == 0.:
            explained_variance = 0.0
        else:
            explained_variance =  1. - (numerator / denominator)
            explained_variance = explained_variance.item()
        return explained_variance


class InteractionTensor(BaseTensor):
    '''4D-Communication Tensor built from gene expression matrices for different contexts
     and a list of ligand-receptor pairs

    Parameters
    ----------
    rnaseq_matrices : list
        A list with dataframes of gene expression wherein the rows are the genes and
        columns the cell types, tissues or samples.

    ppi_data : pandas.DataFrame
        A dataframe containing protein-protein interactions (rows). It has to
        contain at least two columns, one for the first protein partner in the
        interaction as well as the second protein partner.

    order_labels : list, default=None
        List containing the labels for each order or dimension of the tensor. For
        example: ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

    context_names : list, default=None
        A list of strings containing the names of the corresponding contexts to each
        rnaseq_matrix. The length of this list must match the length of the list
        rnaseq_matrices.

    how : str, default='inner'
        Approach to consider cell types and genes present across multiple contexts.

        - 'inner' : Considers only cell types and genes that are present in all
                    contexts (intersection).
        - 'outer' : Considers all cell types and genes that are present
                    across contexts (union).
        - 'outer_genes' : Considers only cell types that are present in all
                          contexts (intersection), while all genes that are
                          present across contexts (union).
        - 'outer_cells' : Considers only genes that are present in all
                          contexts (intersection), while all cell types that are
                          present across contexts (union).

    outer_fraction : float, default=0.0
        Threshold to filter the elements when `how` includes any outer option.
        Elements with a fraction abundance across samples (in `rnaseq_matrices`)
        at least this threshold will be included. When this value is 0, considers
        all elements across the samples. When this value is 1, it acts as using
        `how='inner'`.

    communication_score : str, default='expression_mean'
        Type of communication score to infer the potential use of a given ligand-
        receptor pair by a pair of cells/tissues/samples.
        Available communication_scores are:

        - 'expression_mean' : Computes the average between the expression of a ligand
                              from a sender cell and the expression of a receptor on a
                              receiver cell.
        - 'expression_product' : Computes the product between the expression of a
                                ligand from a sender cell and the expression of a
                                receptor on a receiver cell.
        - 'expression_gmean' : Computes the geometric mean between the expression
                               of a ligand from a sender cell and the
                               expression of a receptor on a receiver cell.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    complex_agg_method : str, default='min'
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.
        - 'gmean' : Geometric mean expression value among all genes.

    upper_letter_comparison : boolean, default=True
        Whether making uppercase the gene names in the expression matrices and the
        protein names in the ppi_data to match their names and integrate their
        respective expression level. Useful when there are inconsistencies in the
        names between the expression matrix and the ligand-receptor annotations.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a dataframe of
        protein-protein interactions. If the list is for ligand-receptor pairs, the
        first column is for the ligands and the second for the receptors.

    group_ppi_by : str, default=None
        Column name in the list of PPIs used for grouping individual PPIs into major
        groups such as signaling pathways.

    group_ppi_method : str, default='gmean'
        Method for aggregating multiple PPIs into major groups.

        - 'mean' : Computes the average communication score among all PPIs of the
                   group for a given pair of cells/tissues/samples
        - 'gmean' : Computes the geometric mean of the communication scores among all
                    PPIs of the group for a given pair of cells/tissues/samples
        - 'sum' : Computes the sum of the communication scores among all PPIs of the
                  group for a given pair of cells/tissues/samples

    device : str, default=None
        Device to use when backend allows using multiple devices. Options are:
         {'cpu', 'cuda:0', None}

    verbose : boolean, default=False
            Whether printing or not steps of the analysis.
    '''
    def __init__(self, rnaseq_matrices, ppi_data, order_labels=None, context_names=None, how='inner', outer_fraction=0.0,
                 communication_score='expression_mean', complex_sep=None, complex_agg_method='min',
                 upper_letter_comparison=True, interaction_columns=('A', 'B'), group_ppi_by=None,
                 group_ppi_method='gmean', device=None, verbose=True):
        # Asserts
        if group_ppi_by is not None:
            assert group_ppi_by in ppi_data.columns, "Using {} for grouping PPIs is not possible. Not present among columns in ppi_data".format(group_ppi_by)


        # Init BaseTensor
        BaseTensor.__init__(self)

        # Generate expression values for protein complexes in PPI data
        if complex_sep is not None:
            if verbose:
                print('Getting expression values for protein complexes')
            col_a_genes, complex_a, col_b_genes, complex_b, complexes = get_genes_from_complexes(ppi_data=ppi_data,
                                                                                                 complex_sep=complex_sep,
                                                                                                 interaction_columns=interaction_columns
                                                                                                 )
            mod_rnaseq_matrices = [add_complexes_to_expression(rnaseq, complexes, agg_method=complex_agg_method) for rnaseq in rnaseq_matrices]
        else:
            mod_rnaseq_matrices = [df.copy() for df in rnaseq_matrices]

        # Uppercase for Gene names
        if upper_letter_comparison:
            for df in mod_rnaseq_matrices:
                df.index = [idx.upper() for idx in df.index]

        # Deduplicate gene names
        mod_rnaseq_matrices = [df[~df.index.duplicated(keep='first')] for df in mod_rnaseq_matrices]

        # Get context CCC tensor
        tensor, genes, cells, ppi_names, mask = build_context_ccc_tensor(rnaseq_matrices=mod_rnaseq_matrices,
                                                                         ppi_data=ppi_data,
                                                                         how=how,
                                                                         outer_fraction=outer_fraction,
                                                                         communication_score=communication_score,
                                                                         complex_sep=complex_sep,
                                                                         upper_letter_comparison=upper_letter_comparison,
                                                                         interaction_columns=interaction_columns,
                                                                         group_ppi_by=group_ppi_by,
                                                                         group_ppi_method=group_ppi_method,
                                                                         verbose=verbose)

        # Distinguish NaNs from real zeros
        if mask is None:
            self.loc_nans = np.zeros(tensor.shape, dtype=int)
        else:
            self.loc_nans = np.ones(tensor.shape, dtype=int) - np.array(mask)
        self.loc_zeros = (tensor == 0).astype(int) - self.loc_nans
        self.loc_zeros = (self.loc_zeros > 0).astype(int)

        # Generate names for the elements in each dimension (order) in the tensor
        if context_names is None:
            context_names = ['C-' + str(i) for i in range(1, len(mod_rnaseq_matrices)+1)]
            # for PPIS use ppis, and sender & receiver cells, use cells

        # Save variables for this class
        self.communication_score = communication_score
        self.how = how
        self.outer_fraction = outer_fraction
        if device is None:
            self.tensor = tl.tensor(tensor)
            self.loc_nans = tl.tensor(self.loc_nans)
            self.loc_zeros = tl.tensor(self.loc_zeros)
            self.mask = mask
        else:
            if tl.get_backend() in ['pytorch', 'tensorflow']: # Potential TODO: Include other backends that support different devices
                self.tensor = tl.tensor(tensor, device=device)
                self.loc_nans = tl.tensor(self.loc_nans, device=device)
                self.loc_zeros = tl.tensor(self.loc_zeros, device=device)
                if mask is not None:
                    self.mask = tl.tensor(mask, device=device)
                else:
                    self.mask = mask
            else:
                self.tensor = tl.tensor(tensor)
                self.loc_nans = tl.tensor(self.loc_nans)
                self.loc_zeros = tl.tensor(self.loc_zeros)
                if mask is not None:
                    self.mask = tl.tensor(mask)
                else:
                    self.mask = mask
        self.genes = genes
        self.cells = cells
        self.order_labels = order_labels
        self.order_names = [context_names, ppi_names, self.cells, self.cells]


class PreBuiltTensor(BaseTensor):
    '''Initializes a cell2cell.tensor.BaseTensor with a prebuilt communication tensor

    Parameters
    ----------
    tensor : ndarray list
        Prebuilt tensor. Could be a list of lists, a numpy array or a tensorly.tensor.

    order_names : list
        List of lists containing the string names of each element in each of the
        dimensions or orders in the tensor. For a 4D-Communication tensor, the first
        list should contain the names of the contexts, the second the names of the
        ligand-receptor interactions, the third the names of the sender cells and the
        fourth the names of the receiver cells.

    order_labels : list, default=None
        List containing the labels for each order or dimension of the tensor. For
        example: ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']

    mask : ndarray list, default=None
        Helps avoiding missing values during a tensor factorization. A mask should be
        a boolean array of the same shape as the original tensor and should be 0
        where the values are missing and 1 everywhere else.

    loc_nans : ndarray list, default=None
        An array of shape equal to `tensor` with ones where NaN values were assigned
        when building the tensor. Other values are zeros. It stores the
        location of the NaN values.

    device : str, default=None
        Device to use when backend allows using multiple devices. Options are:
         {'cpu', 'cuda:0', None}
    '''
    def __init__(self, tensor, order_names, order_labels=None, mask=None, loc_nans=None, device=None):
        # Init BaseTensor
        BaseTensor.__init__(self)

        # Initialize tensor
        try:
            context = tl.context(tensor)
        except:
            context = {'dtype': tensor.dtype, 'device' : None}
        tensor = tl.to_numpy(tensor)
        if mask is not None:
            mask = tl.to_numpy(mask)

        # Location of NaNs and zeros
        tmp_nans = (np.isnan(tensor)).astype(int) # Find extra NaNs that were not considered
        if loc_nans is None:
            self.loc_nans = np.zeros(tuple(tensor.shape), dtype=int)
        else:
            assert loc_nans.shape == tensor.shape, "`loc_nans` and `tensor` must be of the same shape"
            self.loc_nans = np.array(loc_nans.copy())
        self.loc_nans = self.loc_nans + tmp_nans
        self.loc_nans = (self.loc_nans > 0).astype(int)

        self.loc_zeros = (np.array(tensor) == 0.).astype(int) - self.loc_nans
        self.loc_zeros = (self.loc_zeros > 0).astype(int)

        # Store tensor
        tensor_ = np.nan_to_num(tensor)
        if device is not None:
            context['device'] = device
        if 'device' not in context.keys():
            self.tensor = tl.tensor(tensor_)
            self.loc_nans = tl.tensor(self.loc_nans)
            self.loc_zeros = tl.tensor(self.loc_zeros)
            if mask is None:
                self.mask = mask
            else:
                self.mask = tl.tensor(mask)
        else:
            self.tensor = tl.tensor(tensor_, device=context['device'])
            self.loc_nans = tl.tensor(self.loc_nans, device=context['device'])
            self.loc_zeros = tl.tensor(self.loc_zeros, device=context['device'])
            if mask is None:
                self.mask = mask
            else:
                self.mask = tl.tensor(mask, device=context['device'])

        self.order_names = order_names
        if order_labels is None:
            self.order_labels = ['Dimension-{}'.format(i + 1) for i in range(len(self.tensor.shape))]
        else:
            self.order_labels = order_labels
        assert len(self.tensor.shape) == len(self.order_labels), "The length of order_labels must match the number of orders/dimensions in the tensor"


def build_context_ccc_tensor(rnaseq_matrices, ppi_data, how='inner', outer_fraction=0.0,
                             communication_score='expression_product', complex_sep=None, upper_letter_comparison=True,
                             interaction_columns=('A', 'B'), group_ppi_by=None, group_ppi_method='gmean', verbose=True):
    '''Builds a 4D-Communication tensor.
    Takes the gene expression matrices and the list of PPIs to compute
    the communication scores between the interacting cells for each PPI.
    This is done for each context.

    Parameters
    ----------
    rnaseq_matrices : list
        A list with dataframes of gene expression wherein the rows are the genes and
        columns the cell types, tissues or samples.

    ppi_data : pandas.DataFrame
        A dataframe containing protein-protein interactions (rows). It has to
        contain at least two columns, one for the first protein partner in the
        interaction as well as the second protein partner.

    how : str, default='inner'
        Approach to consider cell types and genes present across multiple contexts.

        - 'inner' : Considers only cell types and genes that are present in all
                    contexts (intersection).
        - 'outer' : Considers all cell types and genes that are present
                    across contexts (union).
        - 'outer_genes' : Considers only cell types that are present in all
                          contexts (intersection), while all genes that are
                          present across contexts (union).
        - 'outer_cells' : Considers only genes that are present in all
                          contexts (intersection), while all cell types that are
                          present across contexts (union).

    outer_fraction : float, default=0.0
        Threshold to filter the elements when `how` includes any outer option.
        Elements with a fraction abundance across samples (in `rnaseq_matrices`)
        at least this threshold will be included. When this value is 0, considers
        all elements across the samples. When this value is 1, it acts as using
        `how='inner'`.

    communication_score : str, default='expression_mean'
        Type of communication score to infer the potential use of a given ligand-
        receptor pair by a pair of cells/tissues/samples.
        Available communication_scores are:

        - 'expression_mean' : Computes the average between the expression of a ligand
                              from a sender cell and the expression of a receptor on a
                              receiver cell.
        - 'expression_product' : Computes the product between the expression of a
                                ligand from a sender cell and the expression of a
                                receptor on a receiver cell.
        - 'expression_gmean' : Computes the geometric mean between the expression
                               of a ligand from a sender cell and the
                               expression of a receptor on a receiver cell.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    upper_letter_comparison : boolean, default=True
        Whether making uppercase the gene names in the expression matrices and the
        protein names in the ppi_data to match their names and integrate their
        respective expression level. Useful when there are inconsistencies in the
        names between the expression matrix and the ligand-receptor annotations.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a dataframe of
        protein-protein interactions. If the list is for ligand-receptor pairs, the
        first column is for the ligands and the second for the receptors.

    group_ppi_by : str, default=None
        Column name in the list of PPIs used for grouping individual PPIs into major
        groups such as signaling pathways.

    group_ppi_method : str, default='gmean'
        Method for aggregating multiple PPIs into major groups.

        - 'mean' : Computes the average communication score among all PPIs of the
                   group for a given pair of cells/tissues/samples
        - 'gmean' : Computes the geometric mean of the communication scores among all
                    PPIs of the group for a given pair of cells/tissues/samples
        - 'sum' : Computes the sum of the communication scores among all PPIs of the
                  group for a given pair of cells/tissues/samples

    verbose : boolean, default=False
            Whether printing or not steps of the analysis.

    Returns
    -------
    tensors : list
        List of 3D-Communication tensors for each context. This list corresponds to
        the 4D-Communication tensor.

    genes : list
        List of genes included in the tensor.

    cells : list
        List of cells included in the tensor.

    ppi_names: list
        List of names for each of the PPIs included in the tensor. Used as labels for the
        elements in the cognate tensor dimension (in the attribute order_names of the
        InteractionTensor)

    mask_tensor: numpy.array
        Mask used to exclude values in the tensor. When using how='outer' it masks
        missing values (e.g., cell types that are not present in a given context),
        while using how='inner' makes the mask_tensor to be None.
    '''
    df_idxs = [list(rnaseq.index) for rnaseq in rnaseq_matrices]
    df_cols = [list(rnaseq.columns) for rnaseq in rnaseq_matrices]

    if how == 'inner':
        genes = set.intersection(*map(set, df_idxs))
        cells = set.intersection(*map(set, df_cols))
    elif how == 'outer':
        genes = set(get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_idxs),
                                               fraction=outer_fraction))
        cells = set(get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_cols),
                                               fraction=outer_fraction))
    elif how == 'outer_genes':
        genes = set(get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_idxs),
                                               fraction=outer_fraction))
        cells = set.intersection(*map(set, df_cols))
    elif how == 'outer_cells':
        genes = set.intersection(*map(set, df_idxs))
        cells = set(get_elements_over_fraction(abundance_dict=get_element_abundances(element_lists=df_cols),
                                               fraction=outer_fraction))
    else:
        raise ValueError('Provide a valid way to build the tensor; "how" must be "inner", "outer", "outer_genes" or "outer_cells"')

    # Preserve order or sort new set (either inner or outer)
    if set(df_idxs[0]) == genes:
        genes = df_idxs[0]
    else:
        genes = sorted(list(genes))

    if set(df_cols[0]) == cells:
        cells = df_cols[0]
    else:
        cells = sorted(list(cells))

    # Filter PPI data for
    ppi_data_ = filter_ppi_by_proteins(ppi_data=ppi_data,
                                       proteins=genes,
                                       complex_sep=complex_sep,
                                       upper_letter_comparison=upper_letter_comparison,
                                       interaction_columns=interaction_columns)

    if verbose:
        print('Building tensor for the provided context')

    tensors = [generate_ccc_tensor(rnaseq_data=rnaseq.reindex(genes).reindex(cells, axis='columns'),
                                   ppi_data=ppi_data_,
                                   communication_score=communication_score,
                                   interaction_columns=interaction_columns) for rnaseq in rnaseq_matrices]

    if group_ppi_by is not None:
        ppi_names = [group for group, _ in ppi_data_.groupby(group_ppi_by)]
        tensors = [aggregate_ccc_tensor(ccc_tensor=t,
                                        ppi_data=ppi_data_,
                                        group_ppi_by=group_ppi_by,
                                        group_ppi_method=group_ppi_method) for t in tensors]
    else:
        ppi_names = [row[interaction_columns[0]] + '^' + row[interaction_columns[1]] for idx, row in ppi_data_.iterrows()]

    # Generate mask:
    if how != 'inner':
        mask_tensor = (~np.isnan(np.asarray(tensors))).astype(int)
    else:
        mask_tensor = None
    tensors = np.nan_to_num(tensors)
    return tensors, genes, cells, ppi_names, mask_tensor


def generate_ccc_tensor(rnaseq_data, ppi_data, communication_score='expression_product', interaction_columns=('A', 'B')):
    '''Computes a 3D-Communication tensor for a given context based on the gene
    expression matrix and the list of PPIS

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression matrix for a given context, sample or condition. Rows are
        genes and columns are cell types/tissues/samples.

    ppi_data : pandas.DataFrame
        A dataframe containing protein-protein interactions (rows). It has to
        contain at least two columns, one for the first protein partner in the
        interaction as well as the second protein partner.

    communication_score : str, default='expression_mean'
        Type of communication score to infer the potential use of a given ligand-
        receptor pair by a pair of cells/tissues/samples.
        Available communication_scores are:

        - 'expression_mean' : Computes the average between the expression of a ligand
                              from a sender cell and the expression of a receptor on a
                              receiver cell.
        - 'expression_product' : Computes the product between the expression of a
                                ligand from a sender cell and the expression of a
                                receptor on a receiver cell.
        - 'expression_gmean' : Computes the geometric mean between the expression
                               of a ligand from a sender cell and the
                               expression of a receptor on a receiver cell.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a dataframe of
        protein-protein interactions. If the list is for ligand-receptor pairs, the
        first column is for the ligands and the second for the receptors.

    Returns
    -------
    ccc_tensor : ndarray list
        List of directed cell-cell communication matrices, one for each ligand-
        receptor pair in ppi_data. These matrices contain the communication score for
        pairs of cells for the corresponding PPI. This tensor represent a
        3D-communication tensor for the context.
    '''
    ppi_a = interaction_columns[0]
    ppi_b = interaction_columns[1]

    ccc_tensor = []
    for idx, ppi in ppi_data.iterrows():
        v = rnaseq_data.loc[ppi[ppi_a], :].values
        w = rnaseq_data.loc[ppi[ppi_b], :].values
        ccc_tensor.append(compute_ccc_matrix(prot_a_exp=v,
                                             prot_b_exp=w,
                                             communication_score=communication_score).tolist())
    return ccc_tensor


def aggregate_ccc_tensor(ccc_tensor, ppi_data, group_ppi_by=None, group_ppi_method='gmean'):
    '''Aggregates communication scores of multiple PPIs into major groups
    (e.g., pathways) in a communication tensor

    Parameters
    ----------
    ccc_tensor : ndarray list
        List of directed cell-cell communication matrices, one for each ligand-
        receptor pair in ppi_data. These matrices contain the communication score for
        pairs of cells for the corresponding PPI. This tensor represent a
        3D-communication tensor for the context.

    ppi_data : pandas.DataFrame
        A dataframe containing protein-protein interactions (rows). It has to
        contain at least two columns, one for the first protein partner in the
        interaction as well as the second protein partner.

    group_ppi_by : str, default=None
        Column name in the list of PPIs used for grouping individual PPIs into major
        groups such as signaling pathways.

    group_ppi_method : str, default='gmean'
        Method for aggregating multiple PPIs into major groups.

        - 'mean' : Computes the average communication score among all PPIs of the
                   group for a given pair of cells/tissues/samples
        - 'gmean' : Computes the geometric mean of the communication scores among all
                    PPIs of the group for a given pair of cells/tissues/samples
        - 'sum' : Computes the sum of the communication scores among all PPIs of the
                  group for a given pair of cells/tissues/samples

    Returns
    -------
    aggregated_tensor : ndarray list
        List of directed cell-cell communication matrices, one for each major group of
        ligand-receptor pair in ppi_data. These matrices contain the communication
        score for pairs of cells for the corresponding PPI group. This tensor
        represent a 3D-communication tensor for the context, but for major groups
        instead of individual PPIs.
    '''
    tensor_ = np.array(ccc_tensor)
    aggregated_tensor = []
    for group, df in ppi_data.groupby(group_ppi_by):
        lr_idx = list(df.index)
        ccc_matrices = tensor_[lr_idx]
        aggregated_tensor.append(aggregate_ccc_matrices(ccc_matrices=ccc_matrices,
                                                        method=group_ppi_method).tolist())
    return aggregated_tensor


def generate_tensor_metadata(interaction_tensor, metadata_dicts, fill_with_order_elements=True):
    '''Uses a list of of dicts (or None when a dict is missing) to generate a list of
    metadata for each order in the tensor.

    Parameters
    ----------
    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor.

    metadata_dicts : list
        A list of dictionaries. Each dictionary represents an order of the tensor. In
        an interaction tensor these orders should be contexts, LR pairs, sender cells
        and receiver cells. The keys are the elements in each order (they are
        contained in interaction_tensor.order_names) and the values are the categories
        that each elements will be assigned as metadata.

    fill_with_order_elements : boolean, default=True
        Whether using each element of a dimension as its own metadata when a None is
        passed instead of a dictionary for the respective order/dimension. If True,
        each element in that order will be use itself, that dimension will not contain
        metadata.

    Returns
    -------
    metadata : list
        A list of pandas.DataFrames that will be used as an input of the
        cell2cell.plot.tensor_factors_plot.
    '''
    tensor_dim = len(interaction_tensor.tensor.shape)
    assert (tensor_dim == len(metadata_dicts)), "metadata_dicts should be of the same size as the number of orders/dimensions in the tensor"

    if interaction_tensor.order_labels is None:
        if tensor_dim == 4:
            default_cats = ['Contexts', 'Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
        elif tensor_dim > 4:
            default_cats = ['Contexts-{}'.format(i + 1) for i in range(tensor_dim - 3)] + ['Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
        elif tensor_dim == 3:
            default_cats = ['Ligand-Receptor Pairs', 'Sender Cells', 'Receiver Cells']
        else:
            raise ValueError('Too few dimensions in the tensor')
    else:
        assert len(interaction_tensor.order_labels) == tensor_dim, "The length of order_labels must match the number of orders/dimensions in the tensor"
        default_cats = interaction_tensor.order_labels

    if fill_with_order_elements:
        metadata = [pd.DataFrame(index=names) for names in interaction_tensor.order_names]
    else:
        metadata = [pd.DataFrame(index=names) if (meta is not None) else None for names, meta in zip(interaction_tensor.order_names, metadata_dicts)]

    for i, meta in enumerate(metadata):
        if meta is not None:
            if metadata_dicts[i] is not None:
                meta['Category'] = [metadata_dicts[i][idx] for idx in meta.index]
            else: # dict is None and fill with order elements TRUE
                if len(meta.index) < 75:
                    meta['Category'] = meta.index
                else:
                    meta['Category'] = len(meta.index) * [default_cats[i]]
            meta.index.name = 'Element'
            meta.reset_index(inplace=True)
    return metadata


def interactions_to_tensor(interactions, experiment='single_cell', context_names=None, how='inner', outer_fraction=0.0,
                           communication_score='expression_product', upper_letter_comparison=True, verbose=True):
    '''Takes a list of Interaction pipelines (see classes in
    cell2cell.analysis.pipelines) and generates a communication
    tensor.

    Parameters
    ----------
    interactions : list
        List of Interaction pipelines. The Interactions has to be all either
        BulkInteractions or SingleCellInteractions.

    experiment : str, default='single_cell'
        Type of Interaction pipelines in the list. Either 'single_cell' or 'bulk'.

    context_names : list
        List of context names or labels for each of the Interaction pipelines. This
        list matches the length of interactions and the labels have to follows the
        same order.

    how : str, default='inner'
        Approach to consider cell types and genes present across multiple contexts.

        - 'inner' : Considers only cell types and genes that are present in all
                    contexts (intersection).
        - 'outer' : Considers all cell types and genes that are present
                    across contexts (union).
        - 'outer_genes' : Considers only cell types that are present in all
                          contexts (intersection), while all genes that are
                          present across contexts (union).
        - 'outer_cells' : Considers only genes that are present in all
                          contexts (intersection), while all cell types that are
                          present across contexts (union).

    outer_fraction : float, default=0.0
        Threshold to filter the elements when `how` includes any outer option.
        Elements with a fraction abundance across samples at least this
        threshold will be included. When this value is 0, considers
        all elements across the samples. When this value is 1, it acts as using
        `how='inner'`.

    communication_score : str, default='expression_mean'
        Type of communication score to infer the potential use of a given ligand-
        receptor pair by a pair of cells/tissues/samples.
        Available communication_scores are:

        - 'expression_mean' : Computes the average between the expression of a ligand
                              from a sender cell and the expression of a receptor on a
                              receiver cell.
        - 'expression_product' : Computes the product between the expression of a
                                ligand from a sender cell and the expression of a
                                receptor on a receiver cell.
        - 'expression_gmean' : Computes the geometric mean between the expression
                               of a ligand from a sender cell and the
                               expression of a receptor on a receiver cell.

    upper_letter_comparison : boolean, default=True
        Whether making uppercase the gene names in the expression matrices and the
        protein names in the ppi_data to match their names and integrate their
        respective expression level. Useful when there are inconsistencies in the
        names between the expression matrix and the ligand-receptor annotations.

    Returns
    -------
    tensor : cell2cell.tensor.InteractionTensor
        A 4D-communication tensor.

    '''
    ppis = []
    rnaseq_matrices = []
    complex_sep = interactions[0].complex_sep
    complex_agg_method = interactions[0].complex_agg_method
    interaction_columns = interactions[0].interaction_columns
    for Int_ in interactions:
        if Int_.analysis_setup['cci_type'] == 'undirected':
            ppis.append(Int_.ref_ppi)
        else:
            ppis.append(Int_.ppi_data)

        if experiment == 'single_cell':
            rnaseq_matrices.append(Int_.aggregated_expression)
        elif experiment == 'bulk':
            rnaseq_matrices.append(Int_.rnaseq_data)
        else:
            raise ValueError("experiment must be 'single_cell' or 'bulk'")

    ppi_data = pd.concat(ppis)
    ppi_data = ppi_data.drop_duplicates().reset_index(drop=True)

    tensor = InteractionTensor(rnaseq_matrices=rnaseq_matrices,
                               ppi_data=ppi_data,
                               context_names=context_names,
                               how=how,
                               outer_fraction=outer_fraction,
                               complex_sep=complex_sep,
                               complex_agg_method=complex_agg_method,
                               interaction_columns=interaction_columns,
                               communication_score=communication_score,
                               upper_letter_comparison=upper_letter_comparison,
                               verbose=verbose
                               )
    return tensor