# -*- coding: utf-8 -*-

from __future__ import absolute_import

import tensorly as tl

from cell2cell.plotting.tensor_plot import tensor_factors_plot


def run_tensor_cell2cell_pipeline(interaction_tensor, tensor_metadata, copy_tensor=False, rank=None,
                                  tf_optimization='regular', random_state=None, backend=None, device=None,
                                  elbow_metric='error', smooth_elbow=False, upper_rank=25, tf_init='random',
                                  tf_svd='numpy_svd', cmaps=None, sample_col='Element', group_col='Category',
                                  fig_fontsize=14, output_folder=None, output_fig=True, fig_format='pdf'):
    '''
    Runs basic pipeline of Tensor-cell2cell (excluding downstream analyses).

    Parameters
    ----------
    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor generated with any of the tensor class in
        cell2cell.tensor.

    tensor_metadata : list
        List of pandas dataframes with metadata information for elements of each
        dimension in the tensor. A column called as the variable `sample_col` contains
        the name of each element in the tensor while another column called as the
        variable `group_col` contains the metadata or grouping information of each
        element.

    copy_tensor : boolean, default=False
        Whether generating a copy of the original tensor to avoid modifying it.

    rank : int, default=None
        Rank of the Tensor Factorization (number of factors to deconvolve the original
        tensor). If None, it will automatically inferred from an elbow analysis.

    tf_optimization : str, default='regular'
        It defines whether performing an optimization with higher number of iterations,
        independent factorization runs, and higher resolution (lower tolerance),
        or with lower number of iterations, factorization runs, and resolution.
        Options are:

        - 'regular' : It uses 100 max iterations, 1 factorization run, and 10e-7 tolerance.
                      Faster to run.
        - 'robust' : It uses 500 max iterations, 100 factorization runs, and 10e-8 tolerance.
                     Slower to run.

    random_state : boolean, default=None
        Seed for randomization.

    backend : str, default=None
        Backend that TensorLy will use to perform calculations
        on this tensor. When None, the default backend used is
        the currently active backend, usually is ('numpy'). Options are:
        {'cupy', 'jax', 'mxnet', 'numpy', 'pytorch', 'tensorflow'}

    device : str, default=None
        Device to use when backend allows multiple devices. Options are:
         {'cpu', 'cuda:0', None}

    elbow_metric : str, default='error'
        Metric to perform the elbow analysis (y-axis).

            - 'error' : Normalized error to compute the elbow.
            - 'similarity' : Similarity based on CorrIndex (1-CorrIndex).

    smooth_elbow : boolean, default=False
        Whether smoothing the elbow-analysis curve with a Savitzky-Golay filter.

    upper_rank : int, default=25
        Upper bound of ranks to explore with the elbow analysis.

    tf_init : str, default='random'
        Initialization method for computing the Tensor Factorization.
        {‘svd’, ‘random’}

    tf_svd : str, default='numpy_svd'
        Function to compute the SVD for initializing the Tensor Factorization,
        acceptable values in tensorly.SVD_FUNS

    cmaps : list, default=None
        A list of colormaps used for coloring elements in each dimension. The length
        of this list is equal to the number of dimensions of the tensor. If None, all
        dimensions will be colores with the colormap 'gist_rainbow'.

    sample_col : str, default='Element'
        Name of the column containing the element names in the metadata.

    group_col : str, default='Category'
        Name of the column containing the metadata or grouping information for each
        element in the metadata.

    fig_fontsize : int, default=14
        Font size of the tick labels. Axis labels will be 1.2 times the fontsize.

    output_folder : str, default=None
        Path to the folder where the figures generated will be saved.
        If None, figures will not be saved.

    output_fig : boolean, default=True
        Whether generating the figures with matplotlib.

    fig_format : str, default='pdf'
        Format to store figures when an `output_folder` is specified
        and `output_fig` is True. Otherwise, this is not necessary.

    Returns
    -------
    interaction_tensor : cell2cell.tensor.tensor.BaseTensor
        Either the original input `interaction_tensor` or a copy of it.
        This also stores the results from running the Tensor-cell2cell
        pipeline in the corresponding attributes.
    '''
    if copy_tensor:
        interaction_tensor = interaction_tensor.copy()

    dim = len(interaction_tensor.tensor.shape)

    ### OUTPUT FILENAMES ###
    if output_folder is None:
        elbow_filename = None
        tf_filename = None
        loading_filename = None
    else:
        elbow_filename = output_folder + '/Elbow.{}'.format(fig_format)
        tf_filename = output_folder + '/Tensor-Factorization.{}'.format(fig_format)
        loading_filename = output_folder + '/Loadings.xlsx'

    ### PALETTE COLORS FOR ELEMENTS IN TENSOR DIMS ###
    if cmaps is None:
        cmap_5d = ['tab10', 'viridis', 'Dark2_r', 'tab20', 'tab20']
        cmap_4d = ['plasma', 'Dark2_r', 'tab20', 'tab20']

        if dim == 5:
            cmaps = cmap_5d
        elif dim <= 4:
            cmaps = cmap_4d[-dim:]
        else:
            raise ValueError('Tensor of dimension higher to 5 is not supported')

    assert len(cmaps) == dim, "`cmap` must be of the same len of dimensions in the tensor."

    ### FACTORIZATION PARAMETERS ###
    if tf_optimization == 'robust':
        elbow_runs = 20
        tf_runs = 100
        tol = 1e-8
        n_iter_max = 500
    elif tf_optimization == 'regular':
        elbow_runs = 10
        tf_runs = 1
        tol = 1e-7
        n_iter_max = 100
    else:
        raise ValueError("`factorization_type` must be either 'robust' or 'regular'.")

    if backend is not None:
        tl.set_backend(backend)

    if device is not None:
        try:
            interaction_tensor.tensor = tl.tensor(interaction_tensor.tensor, device=device)
            if interaction_tensor.mask is not None:
                interaction_tensor.mask = tl.tensor(interaction_tensor.mask, device=device)
        except:
            interaction_tensor.tensor = tl.tensor(interaction_tensor.tensor)
            if interaction_tensor.mask is not None:
                interaction_tensor.mask = tl.tensor(interaction_tensor.mask)

    ### ANALYSIS ###
    # Elbow
    if rank is None:
        print('Running Elbow Analysis')
        fig1, error = interaction_tensor.elbow_rank_selection(upper_rank=upper_rank,
                                                              runs=elbow_runs,
                                                              init=tf_init,
                                                              svd=tf_svd,
                                                              automatic_elbow=True,
                                                              metric=elbow_metric,
                                                              output_fig=output_fig,
                                                              smooth=smooth_elbow,
                                                              random_state=random_state,
                                                              fontsize=fig_fontsize,
                                                              filename=elbow_filename,
                                                              tol=tol, n_iter_max=n_iter_max
                                                              )

        rank = interaction_tensor.rank

    # Factorization
    print('Running Tensor Factorization')
    interaction_tensor.compute_tensor_factorization(rank=rank,
                                                    init=tf_init,
                                                    svd=tf_svd,
                                                    random_state=random_state,
                                                    runs=tf_runs,
                                                    normalize_loadings=True,
                                                    tol=tol, n_iter_max=n_iter_max
                                                    )

    ### EXPORT RESULTS ###
    if output_folder is not None:
        print('Generating Outputs')
        interaction_tensor.export_factor_loadings(loading_filename)

    if output_fig:
        fig2, axes = tensor_factors_plot(interaction_tensor=interaction_tensor,
                                         metadata=tensor_metadata,
                                         sample_col=sample_col,
                                         group_col=group_col,
                                         meta_cmaps=cmaps,
                                         fontsize=fig_fontsize,
                                         filename=tf_filename
                                         )

    return interaction_tensor