# -*- coding: utf-8 -*-

'''Tests for cell2cell.spatial'''

import numpy as np
import pandas as pd
import pytest

import cell2cell as c2c
from cell2cell.spatial import distances, filtering, neighborhoods


# ---------------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------------

def test_celltype_pair_distance_min():
    df1 = pd.DataFrame({'X': [0.0, 1.0], 'Y': [0.0, 0.0]})
    df2 = pd.DataFrame({'X': [3.0, 10.0], 'Y': [0.0, 0.0]})
    assert np.isclose(distances.celltype_pair_distance(df1, df2, method='min'), 2.0)


def test_celltype_pair_distance_max():
    df1 = pd.DataFrame({'X': [0.0], 'Y': [0.0]})
    df2 = pd.DataFrame({'X': [3.0, 4.0], 'Y': [0.0, 0.0]})
    assert np.isclose(distances.celltype_pair_distance(df1, df2, method='max'), 4.0)


def test_celltype_pair_distance_mean():
    df1 = pd.DataFrame({'X': [0.0], 'Y': [0.0]})
    df2 = pd.DataFrame({'X': [2.0, 4.0], 'Y': [0.0, 0.0]})
    assert np.isclose(distances.celltype_pair_distance(df1, df2, method='mean'), 3.0)


def test_celltype_pair_distance_manhattan():
    df1 = pd.DataFrame({'X': [0.0], 'Y': [0.0]})
    df2 = pd.DataFrame({'X': [3.0], 'Y': [4.0]})
    euclidean = distances.celltype_pair_distance(df1, df2, distance='euclidean')
    manhattan = distances.celltype_pair_distance(df1, df2, distance='manhattan')
    assert np.isclose(euclidean, 5.0)
    assert np.isclose(manhattan, 7.0)


def test_celltype_pair_distance_of_a_cluster_with_itself_is_zero():
    df = pd.DataFrame({'X': [0.0, 1.0], 'Y': [0.0, 1.0]})
    assert np.isclose(distances.celltype_pair_distance(df, df, method='min'), 0.0)


def test_celltype_pair_distance_rejects_bad_options():
    df = pd.DataFrame({'X': [0.0], 'Y': [0.0]})
    with pytest.raises(NotImplementedError):
        distances.celltype_pair_distance(df, df, method='nonsense')
    with pytest.raises(NotImplementedError):
        distances.celltype_pair_distance(df, df, distance='nonsense')


def test_pairwise_celltype_distances_is_a_distance_matrix(toy_coordinates):
    result = distances.pairwise_celltype_distances(toy_coordinates, group_col='celltype',
                                                   coord_cols=['X', 'Y'])
    assert list(result.index) == list(result.columns)
    assert set(result.index) == {'CT-1', 'CT-2', 'CT-3'}
    assert np.allclose(np.diag(result.values), 0.0)
    assert np.allclose(result.values, result.values.T)
    assert (result.values >= 0).all()


def test_pairwise_celltype_distances_matches_the_pairwise_function(toy_coordinates):
    result = distances.pairwise_celltype_distances(toy_coordinates, group_col='celltype')
    group_a = toy_coordinates[toy_coordinates['celltype'] == 'CT-1'][['X', 'Y']]
    group_b = toy_coordinates[toy_coordinates['celltype'] == 'CT-2'][['X', 'Y']]
    expected = distances.celltype_pair_distance(group_a, group_b, method='min')
    assert np.isclose(result.loc['CT-1', 'CT-2'], expected)


def test_pairwise_celltype_distances_with_explicit_pairs(toy_coordinates):
    result = distances.pairwise_celltype_distances(toy_coordinates, group_col='celltype',
                                                   pairs=[('CT-1', 'CT-2')])
    assert result.loc['CT-1', 'CT-2'] > 0


# ---------------------------------------------------------------------------------
# neighborhoods
# ---------------------------------------------------------------------------------

def test_create_spatial_grid_adds_columns(toy_spatial_adata):
    c2c.spatial.create_spatial_grid(toy_spatial_adata, num_bins=5)
    for column in ['grid_x', 'grid_y', 'grid_cell']:
        assert column in toy_spatial_adata.obs.columns
    assert toy_spatial_adata.obs['grid_x'].min() >= 0
    assert toy_spatial_adata.obs['grid_x'].max() <= 4


def test_create_spatial_grid_copy_does_not_modify_the_original(toy_spatial_adata):
    result = c2c.spatial.create_spatial_grid(toy_spatial_adata, num_bins=4, copy=True)
    assert 'grid_cell' in result.obs.columns
    assert 'grid_cell' not in toy_spatial_adata.obs.columns


def test_create_spatial_grid_cell_ids_combine_both_axes(toy_spatial_adata):
    c2c.spatial.create_spatial_grid(toy_spatial_adata, num_bins=4)
    for value in toy_spatial_adata.obs['grid_cell']:
        assert '_' in value
    grid = toy_spatial_adata.obs
    rebuilt = grid['grid_x'].astype(str) + '_' + grid['grid_y'].astype(str)
    assert (rebuilt == grid['grid_cell']).all()


def test_calculate_window_size(toy_spatial_adata):
    size = c2c.spatial.calculate_window_size(toy_spatial_adata, num_windows=4)
    coords = toy_spatial_adata.obsm['spatial'][:, 0]
    assert np.isclose(size, (coords.max() - coords.min()) / 4)


def test_create_sliding_windows_maps_barcodes(toy_spatial_adata):
    mapping = c2c.spatial.create_sliding_windows(toy_spatial_adata, window_size=25.,
                                                 stride=25.)
    assert len(mapping) > 0
    barcodes = set(toy_spatial_adata.obs_names)
    for window, members in mapping.items():
        assert window.startswith('window_')
        assert isinstance(members, set)
        assert members.issubset(barcodes)


def test_create_sliding_windows_covers_the_interior(toy_spatial_adata):
    '''Documents a boundary behaviour: windows are half-open intervals
    [edge, edge + window_size), and the last edge stops before the maximum
    coordinate, so cells sitting exactly on the maximum X or Y are not assigned
    to any window.
    '''
    mapping = c2c.spatial.create_sliding_windows(toy_spatial_adata, window_size=30.,
                                                 stride=10.)
    covered = set().union(*mapping.values())
    coordinates = toy_spatial_adata.obsm['spatial']
    on_the_edge = {barcode for barcode, (x, y)
                   in zip(toy_spatial_adata.obs_names, coordinates)
                   if x == coordinates[:, 0].max() or y == coordinates[:, 1].max()}

    assert covered == set(toy_spatial_adata.obs_names) - on_the_edge
    assert len(covered) > 0 and len(on_the_edge) > 0


def test_overlapping_windows_share_cells(toy_spatial_adata):
    '''With a stride smaller than the window, a cell belongs to several windows.'''
    mapping = c2c.spatial.create_sliding_windows(toy_spatial_adata, window_size=40.,
                                                 stride=10.)
    counts = {}
    for members in mapping.values():
        for barcode in members:
            counts[barcode] = counts.get(barcode, 0) + 1
    assert max(counts.values()) > 1


def test_add_sliding_window_info_marks_only_members(toy_spatial_adata):
    mapping = c2c.spatial.create_sliding_windows(toy_spatial_adata, window_size=25.,
                                                 stride=25.)
    c2c.spatial.add_sliding_window_info_to_adata(toy_spatial_adata, mapping)
    for window, members in mapping.items():
        column = toy_spatial_adata.obs[window]
        assert set(column.unique()).issubset({0.0, 1.0})
        assert (column.loc[list(members)] == 1.0).all()
        outsiders = set(toy_spatial_adata.obs_names) - members
        if outsiders:
            assert (column.loc[list(outsiders)] == 0.0).all()


# ---------------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------------

@pytest.fixture
def celltype_distances():
    cells = ['CT-1', 'CT-2', 'CT-3']
    values = np.array([[0., 10., 100.],
                       [10., 0., 50.],
                       [100., 50., 0.]])
    return pd.DataFrame(values, index=cells, columns=cells)


@pytest.fixture
def liana_tensor(toy_liana):
    context_dict = {name: frame for name, frame in toy_liana.groupby('context')}
    return c2c.tensor.dataframes_to_tensor(context_dict, sender_col='source',
                                           receiver_col='target', ligand_col='ligand',
                                           receptor_col='receptor', score_col='score',
                                           how='inner')


def test_dist_filter_tensor_masks_distant_pairs(liana_tensor, celltype_distances):
    filtered = filtering.dist_filter_tensor(liana_tensor, celltype_distances,
                                            max_dist=20., source_axis=2, target_axis=3)
    senders = list(filtered.order_names[2])
    receivers = list(filtered.order_names[3])
    data = np.asarray(filtered.tensor)
    for s, sender in enumerate(senders):
        for r, receiver in enumerate(receivers):
            if celltype_distances.loc[sender, receiver] > 20.:
                assert np.allclose(np.nan_to_num(data[:, :, s, r]), 0.0)


def test_dist_filter_tensor_keeps_close_pairs(liana_tensor, celltype_distances):
    original = np.asarray(liana_tensor.tensor).copy()
    filtered = filtering.dist_filter_tensor(liana_tensor, celltype_distances, max_dist=20.)
    senders = list(filtered.order_names[2])
    data = np.asarray(filtered.tensor)
    close = senders.index('CT-1'), senders.index('CT-2')
    assert np.allclose(data[:, :, close[0], close[1]],
                       original[:, :, close[0], close[1]])


def test_dist_filter_tensor_returns_a_copy(liana_tensor, celltype_distances):
    original = np.asarray(liana_tensor.tensor).copy()
    filtering.dist_filter_tensor(liana_tensor, celltype_distances, max_dist=20.)
    assert np.allclose(np.asarray(liana_tensor.tensor), original)


def test_dist_filter_tensor_min_dist(liana_tensor, celltype_distances):
    filtered = filtering.dist_filter_tensor(liana_tensor, celltype_distances,
                                            max_dist=200., min_dist=20.)
    senders = list(filtered.order_names[2])
    data = np.asarray(filtered.tensor)
    # CT-1 <-> CT-2 are 10 apart, below min_dist, so they must be filtered out
    i, j = senders.index('CT-1'), senders.index('CT-2')
    assert np.allclose(np.nan_to_num(data[:, :, i, j]), 0.0)


def test_dist_filter_liana_removes_distant_rows(toy_liana, celltype_distances):
    filtered = filtering.dist_filter_liana(toy_liana, celltype_distances, max_dist=20.)
    assert filtered.shape[0] < toy_liana.shape[0]
    for _, row in filtered.iterrows():
        assert celltype_distances.loc[row['source'], row['target']] <= 20.


def test_dist_filter_liana_can_keep_the_distance_column(toy_liana, celltype_distances):
    filtered = filtering.dist_filter_liana(toy_liana, celltype_distances, max_dist=20.,
                                           keep_dist=True)
    assert 'distance' in filtered.columns
    for _, row in filtered.iterrows():
        assert np.isclose(row['distance'],
                          celltype_distances.loc[row['source'], row['target']])


def test_dist_filter_liana_does_not_modify_input(toy_liana, celltype_distances):
    before = toy_liana.copy()
    filtering.dist_filter_liana(toy_liana, celltype_distances, max_dist=20.)
    pd.testing.assert_frame_equal(toy_liana, before)
