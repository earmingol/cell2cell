# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorly as tl


def dist_filter_tensor(interaction_tensor, distances, max_dist, min_dist=0, source_axis=2, target_axis=3):
    '''
    Filters an Interaction Tensor based on intercellular distances between cell types.

    Parameters
    ----------
    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor generated with any of the tensor class in
        cell2cell.tensor

    distances : pandas.DataFrame
        Square dataframe containing distances between pairs of cell groups. It must contain
        all cell groups that act as sender and receiver cells in the tensor.

    max_dist : float
        The maximum distance between cell pairs to consider them in the interaction tensor.

    min_dist : float, default=0
        The minimum distance between cell pairs to consider them in the interaction tensor.

    source_axis : int, default=2
        The index indicating the axis in the tensor corresponding to sender cells.

    target_axis : int, default=3
        The index indicating the axis in the tensor corresponding to receiver cells.

    Returns
    -------
    new_interaction_tensor : cell2cell.tensor.BaseTensor
        A tensor with communication scores made zero for cell type pairs with intercellular
        distance over the distance threshold.
    '''
    # Evaluate whether we provide distances for all cell types in the tensor
    assert all([cell in distances.index for cell in
                interaction_tensor.order_names[source_axis]]), "Distances not provided for all sender cells"
    assert all([cell in distances.columns for cell in
                interaction_tensor.order_names[target_axis]]), "Distances not provided for all receiver cells"

    source_cell_groups = interaction_tensor.order_names[source_axis]
    target_cell_groups = interaction_tensor.order_names[target_axis]

    # Use only cell types in the tensor
    dist_df = distances.loc[source_cell_groups, target_cell_groups]

    # Filter cell types by intercellular distances
    dist = ((min_dist <= dist_df) & (dist_df <= max_dist)).astype(int).values

    # Mapping what re-arrange should be done to keep the original tensor shape
    tensor_shape = list(interaction_tensor.tensor.shape)
    original_order = list(range(len(tensor_shape)))
    new_order = []

    # Generate template tensor with cells to keep
    template_tensor = dist
    for i, size in enumerate(tensor_shape):
        if (i != source_axis) and (i != target_axis):
            template_tensor = [template_tensor] * size
            new_order.insert(0, i)
    template_tensor = np.array(template_tensor)

    new_order += [source_axis, target_axis]
    changes_needed = [new_order.index(i) for i in original_order]

    # Re-arrange axes by the order
    template_tensor = template_tensor.transpose(changes_needed)

    # Create tensorly object
    template_tensor = tl.tensor(template_tensor, **tl.context(interaction_tensor.tensor))

    assert template_tensor.shape == interaction_tensor.tensor.shape, "Filtering of cells was not properly done. Revise code of this function (template tensor)"

    # tensor = tl.zeros_like(interaction_tensor.tensor, **tl.context(tensor))
    new_interaction_tensor = interaction_tensor.copy()
    new_interaction_tensor.tensor = new_interaction_tensor.tensor * template_tensor
    # Make masked cells by distance to be real zeros
    new_interaction_tensor.loc_zeros = (new_interaction_tensor.tensor == 0).astype(int) - new_interaction_tensor.loc_nans
    return new_interaction_tensor


def dist_filter_liana(liana_outputs, distances, max_dist, min_dist=0, source_col='source', target_col='target',
                      keep_dist=False):
    '''
    Filters a dataframe with outputs from LIANA based on a distance threshold
    defined applied to another dataframe containing distances between cell groups.

    Parameters
    ----------
    liana_outputs : pandas.DataFrame
        Dataframe containing the results from LIANA, where rows are pairs of
        ligand-receptor interactions by pair of source-target cell groups.

    distances : pandas.DataFrame
        Square dataframe containing distances between pairs of cell groups.

    max_dist : float
        The distance threshold used to filter the pairs from the liana_outputs dataframe.

    min_dist : float, default=0
        The minimum distance between cell pairs to consider them in the interaction tensor.

    source_col : str, default='source'
        Column name in both dataframes that represents the source cell groups.

    target_col : str, default='target'
         Column name in both dataframes that represents the target cell groups.

    keep_dist : bool, default=False
        To determine whether to keep the 'distance' column in the filtered output.
        If set to True, the 'distance' column will be retained; otherwise, it will be dropped
        and the LIANA dataframe will contain the original columns.

    Returns
    -------
    filtered_liana_outputs : pandas.DataFrame
        It containing pairs from the liana_outputs dataframe that meet the distance
        threshold criteria.
    '''
    # Convert distances to a long-form dataframe
    distances = distances.stack().reset_index()
    distances.columns = [source_col, target_col, 'distance']

    # Merge the long-form distances DataFrame with pairs_df
    merged_df = liana_outputs.merge(distances, on=[source_col, target_col], how='left')

    # Filter based on the distance threshold
    filtered_liana_outputs = merged_df[(min_dist <= merged_df['distance']) & (merged_df['distance'] <= max_dist)]

    if keep_dist == False:
        filtered_liana_outputs = filtered_liana_outputs.drop(['distance'], axis=1)

    return filtered_liana_outputs


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