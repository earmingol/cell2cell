# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def create_spatial_grid(adata, num_bins, copy=False):
    """
    Segments spatial transcriptomics data into a square grid based on spatial coordinates
    and annotates each cell or spot with its corresponding grid position.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial transcriptomics data. The spatial coordinates
        must be stored in `adata.obsm['spatial']`. This object is either modified in place
        or a copy is returned based on the `copy` parameter.

    num_bins : int
        The number of bins (squares) along each dimension of the grid. The grid is square,
        so this number applies to both the horizontal and vertical divisions.

    copy : bool, default=False
        If True, the function operates on and returns a copy of the input AnnData object.
        If False, the function modifies the input AnnData object in place.

    Returns
    -------
    adata_ : AnnData or None
        If `copy=True`, a new AnnData object with added grid annotations is returned.
    """

    if copy:
        adata_ = adata.copy()
    else:
        adata_ = adata

    # Get the spatial coordinates
    coords = pd.DataFrame(adata.obsm['spatial'], index=adata.obs_names, columns=['X', 'Y'])

    # Define the bins for each dimension
    x_min, y_min = coords.min()
    x_max, y_max = coords.max()
    x_bins = np.linspace(x_min, x_max, num_bins + 1)
    y_bins = np.linspace(y_min, y_max, num_bins + 1)

    # Digitize the coordinates into bins
    adata_.obs['grid_x'] = np.digitize(coords['X'], x_bins, right=False) - 1
    adata_.obs['grid_y'] = np.digitize(coords['Y'], y_bins, right=False) - 1

    # Adjust indices to start from 0 and end at num_bins - 1
    adata_.obs['grid_x'] = np.clip(adata_.obs['grid_x'], 0, num_bins - 1)
    adata_.obs['grid_y'] = np.clip(adata_.obs['grid_y'], 0, num_bins - 1)

    # Combine grid indices to form a grid cell identifier
    adata_.obs['grid_cell'] = adata_.obs['grid_x'].astype(str) + "_" + adata_.obs['grid_y'].astype(str)

    if copy:
        return adata_


def calculate_window_size(adata, num_windows):
    """
    Calculates the window size required to fit a specified number of windows
    across the width of the coordinate space in spatial transcriptomics data.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial transcriptomics data. The spatial coordinates
        must be stored in `adata.obsm['spatial']`.

    num_windows : int
        The desired number of windows to fit across the width of the coordinate space.

    Returns
    -------
    window_size : float
        The calculated size of each window to fit the specified number of windows
        across the width of the coordinate space.
    """

    # Extract X coordinates
    x_coords = adata.obsm['spatial'][:, 0]

    # Determine the range of X coordinates
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    # Calculate the window size
    window_size = (x_max - x_min) / num_windows

    return window_size


def create_sliding_windows(adata, window_size, stride):
    """
    Maps windows to the cells they contain based on spatial transcriptomics data.
    Returns a dictionary where keys are window identifiers and values are sets of cell indices.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial transcriptomics data. The spatial coordinates
        must be stored in `adata.obsm['spatial']`.

    window_size : float
        The size of each square window along each dimension.

    stride : float
        The stride with which the window moves along each dimension.

    Returns
    -------
    window_mapping : dict
        A dictionary mapping each window to a set of cell indices that fall within that window.
    """

    # Get the spatial coordinates
    coords = pd.DataFrame(adata.obsm['spatial'], index=adata.obs_names, columns=['X', 'Y'])

    # Define the range of the sliding windows
    x_min, y_min = coords.min()
    x_max, y_max = coords.max()
    x_windows = np.arange(x_min, x_max - window_size + stride, stride)
    y_windows = np.arange(y_min, y_max - window_size + stride, stride)

    # Function to find all windows a point belongs to
    def find_windows(coord, window_edges):
        return [i for i, edge in enumerate(window_edges) if edge <= coord < edge + window_size]

    # Initialize the window mapping
    window_mapping = {}

    # Assign cells to all overlapping windows
    for cell_idx, (x, y) in enumerate(zip(coords['X'], coords['Y'])):
        cell_windows = ["window_{}_{}".format(wx, wy)
                        for wx in find_windows(x, x_windows)
                        for wy in find_windows(y, y_windows)]

        for win in cell_windows:
            if win not in window_mapping:
                window_mapping[win] = set()
            window_mapping[win].add(coords.index[cell_idx])  # This stores the cell/spot barcodes
            # For memory efficiency, it could be `window_mapping[win].add(cell_idx)` instead

    return window_mapping


def add_sliding_window_info_to_adata(adata, window_mapping):
    """
    Adds window information to the AnnData object's .obs DataFrame. Each window is represented
    as a column, and cells/spots belonging to a window are marked with a 1.0, while others are marked
    with a 0.0. It modifies the `adata` object in place.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to which the window information will be added.

    window_mapping : dict
        A dictionary mapping each window to a set of cell/spot indeces or barcodes.
        This is the output from the `create_moving_windows` function.
    """

    # Initialize all window columns to 0.0
    for window in sorted(window_mapping.keys()):
        adata.obs[window] = 0.0

    # Mark cells that belong to each window
    for window, barcode_indeces in window_mapping.items():
        adata.obs.loc[barcode_indeces, window] = 1.0