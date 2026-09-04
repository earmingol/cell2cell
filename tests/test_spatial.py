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


def test_pairwise_celltype_distances_leaves_uncomputed_pairs_missing(toy_coordinates):
    '''A pair that was never computed is NaN, not zero.

    Zero is a distance -- two cell groups in the same place -- so it cannot double as
    "not calculated" without making the two indistinguishable.
    '''
    result = distances.pairwise_celltype_distances(toy_coordinates, group_col='celltype',
                                                   pairs=[('CT-1', 'CT-2')])
    assert np.isfinite(result.loc['CT-1', 'CT-2'])
    assert np.isnan(result.loc['CT-1', 'CT-3'])
    assert np.isnan(result.loc['CT-2', 'CT-3'])
    assert np.allclose(np.diag(result.values), 0.0)


@pytest.mark.parametrize('method', ['min', 'max', 'mean', 'median'])
def test_pairwise_celltype_distances_keeps_a_zero_diagonal_for_explicit_self_pairs(
        toy_coordinates, method):
    '''A group is at distance zero from itself, not at the spread of its own cells.'''
    result = distances.pairwise_celltype_distances(toy_coordinates, group_col='celltype',
                                                   method=method,
                                                   pairs=[('CT-1', 'CT-1'),
                                                          ('CT-1', 'CT-2')])
    assert result.loc['CT-1', 'CT-1'] == 0.0


def test_pairwise_celltype_distances_is_naturally_sorted():
    '''Every method must agree on the order of rows and columns.'''
    coordinates = pd.DataFrame({
        'X': [0.0, 1.0, 2.0, 3.0],
        'Y': [0.0, 1.0, 2.0, 3.0],
        'celltype': ['CT-10', 'CT-2', 'CT-1', 'CT-10'],
    })
    expected = ['CT-1', 'CT-2', 'CT-10']
    for method in ('min', 'max', 'mean', 'median'):
        result = distances.pairwise_celltype_distances(coordinates, group_col='celltype',
                                                       method=method)
        assert list(result.index) == expected
        assert list(result.columns) == expected


def test_celltype_distances_orders_every_method_the_same_way():
    coordinates = pd.DataFrame({
        'X': [0.0, 1.0, 2.0, 3.0],
        'Y': [0.0, 1.0, 2.0, 3.0],
        'celltype': ['CT-10', 'CT-2', 'CT-1', 'CT-10'],
    })
    orders = [list(c2c.spatial.celltype_distances(coordinates, group_col='celltype',
                                                 method=method).index)
              for method in ('centroid', 'min', 'max', 'mean', 'median')]
    assert orders == [['CT-1', 'CT-2', 'CT-10']] * len(orders)


def test_check_symmetry_accepts_a_matrix_with_uncomputed_pairs(toy_coordinates):
    '''NaN never equals NaN, so a partial distance matrix needs a NaN-aware check.'''
    partial = distances.pairwise_celltype_distances(toy_coordinates, group_col='celltype',
                                                    pairs=[('CT-1', 'CT-2')])
    assert c2c.preprocessing.check_symmetry(partial)
    c2c.preprocessing.convert_to_distance_matrix(partial)   # must not raise

    asymmetric = partial.copy()
    asymmetric.loc['CT-1', 'CT-2'] = asymmetric.loc['CT-2', 'CT-1'] + 1.0
    assert not c2c.preprocessing.check_symmetry(asymmetric)


def test_correlation_objective_refuses_a_reference_with_uncomputed_pairs():
    '''Correlating against NaN gives NaN, which would silently become zero fitness.'''
    genes = ['G{}'.format(i) for i in range(20)]
    rnaseq = c2c.datasets.generate_random_rnaseq(size=4, row_names=genes, random_state=0,
                                                 verbose=False)
    matrix = np.ones((4, 4))
    np.fill_diagonal(matrix, 0.0)
    matrix[0, 2] = matrix[2, 0] = np.nan
    reference = pd.DataFrame(matrix, index=rnaseq.columns, columns=rnaseq.columns)

    with pytest.raises(ValueError, match='missing values among the cell pairs'):
        c2c.analysis.CorrelationObjective(
            rnaseq_data=rnaseq, reference_distances=reference,
            cutoff_setup={'type': 'constant_value', 'parameter': 10},
            analysis_setup={'communication_score': 'expression_thresholding',
                            'cci_score': 'bray_curtis', 'cci_type': 'undirected'})


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


@pytest.fixture
def partial_celltype_distances(celltype_distances):
    '''Distances with the CT-1/CT-2 pair never computed, as `pairs=` leaves them.'''
    partial = celltype_distances.copy()
    partial.loc['CT-1', 'CT-2'] = np.nan
    partial.loc['CT-2', 'CT-1'] = np.nan
    return partial


def test_dist_filter_tensor_excludes_pairs_with_no_distance(liana_tensor,
                                                            partial_celltype_distances):
    '''An unknown distance cannot pass a threshold, so the pair is masked out.'''
    filtered = filtering.dist_filter_tensor(liana_tensor, partial_celltype_distances,
                                            max_dist=20., source_axis=2, target_axis=3)
    senders = list(filtered.order_names[2])
    receivers = list(filtered.order_names[3])
    data = np.asarray(filtered.tensor)
    i, j = senders.index('CT-1'), receivers.index('CT-2')
    assert np.allclose(np.nan_to_num(data[:, :, i, j]), 0.0)


def test_dist_filter_liana_excludes_pairs_with_no_distance(toy_liana,
                                                           partial_celltype_distances):
    '''Independent of the pandas version, where `stack` differs on missing values.'''
    filtered = filtering.dist_filter_liana(toy_liana, partial_celltype_distances,
                                           max_dist=20.)
    kept = set(zip(filtered['source'], filtered['target']))
    assert ('CT-1', 'CT-2') not in kept
    assert ('CT-2', 'CT-1') not in kept


# ---------------------------------------------------------------------------------
# add_sliding_window_info_to_adata crashed on pandas >= 2
#
# The barcodes of each window were passed to .loc as a set, which newer versions of
# pandas reject with "Passing a set as an indexer is not supported".
# ---------------------------------------------------------------------------------

def test_add_sliding_window_info_to_adata_accepts_window_mapping(toy_spatial_adata):
    window_mapping = c2c.spatial.create_sliding_windows(toy_spatial_adata,
                                                        window_size=20., stride=7.)
    assert isinstance(next(iter(window_mapping.values())), set)

    c2c.spatial.add_sliding_window_info_to_adata(toy_spatial_adata, window_mapping)

    window_columns = [c for c in toy_spatial_adata.obs.columns if c.startswith('window_')]
    assert len(window_columns) == len(window_mapping)
    # Every cell assigned to a window must be flagged with 1.0
    for window, barcodes in window_mapping.items():
        flagged = toy_spatial_adata.obs.loc[list(barcodes), window]
        assert (flagged == 1.0).all()


def test_sliding_window_columns_are_naturally_ordered(toy_spatial_adata):
    '''With more than 10 windows per axis, window_10_* must not precede window_2_*.'''
    window_mapping = c2c.spatial.create_sliding_windows(toy_spatial_adata,
                                                        window_size=10., stride=5.)
    c2c.spatial.add_sliding_window_info_to_adata(toy_spatial_adata, window_mapping)

    columns = [c for c in toy_spatial_adata.obs.columns if c.startswith('window_')]
    assert any(c.startswith('window_10_') for c in columns)
    assert columns != sorted(columns)       # natural order differs from alphabetical


# ---------------------------------------------------------------------------------
# distances from single-cell coordinates in an AnnData object
# ---------------------------------------------------------------------------------

def test_celltype_pair_distance_median():
    df1 = pd.DataFrame({'X': [0.0, 0.0], 'Y': [0.0, 0.0]})
    df2 = pd.DataFrame({'X': [1.0, 3.0], 'Y': [0.0, 0.0]})
    # distances are 1, 3, 1, 3
    assert np.isclose(distances.celltype_pair_distance(df1, df2, method='median'), 2.0)


def test_get_spatial_coordinates(toy_spatial_adata):
    coords = distances.get_spatial_coordinates(toy_spatial_adata)
    assert coords.shape == (toy_spatial_adata.n_obs, 2)
    assert list(coords.columns) == ['X', 'Y']
    assert list(coords.index) == list(toy_spatial_adata.obs_names)
    np.testing.assert_allclose(coords.values, toy_spatial_adata.obsm['spatial'])


def test_get_spatial_coordinates_custom_key(toy_spatial_adata):
    toy_spatial_adata.obsm['X_spatial'] = toy_spatial_adata.obsm['spatial']
    coords = distances.get_spatial_coordinates(toy_spatial_adata, spatial_key='X_spatial')
    np.testing.assert_allclose(coords.values, toy_spatial_adata.obsm['spatial'])


def test_get_spatial_coordinates_missing_key(toy_spatial_adata):
    with pytest.raises(KeyError):
        distances.get_spatial_coordinates(toy_spatial_adata, spatial_key='not_there')


def test_get_spatial_coordinates_custom_names(toy_spatial_adata):
    coords = distances.get_spatial_coordinates(toy_spatial_adata, coord_names=['row', 'col'])
    assert list(coords.columns) == ['row', 'col']


def test_celltype_centroids(toy_spatial_adata):
    centroids = distances.celltype_centroids(toy_spatial_adata, group_col='celltype')
    coords = distances.get_spatial_coordinates(toy_spatial_adata)
    expected = coords.groupby(np.asarray(toy_spatial_adata.obs['celltype'].values)).mean()
    np.testing.assert_allclose(centroids.values, expected.loc[centroids.index].values)


def test_celltype_centroids_are_naturally_sorted():
    rng = np.random.default_rng(0)
    n = 60
    labels = ['CT-{}'.format(i % 12 + 1) for i in range(n)]
    df = pd.DataFrame({'X': rng.random(n), 'Y': rng.random(n), 'celltype': labels})
    centroids = distances.celltype_centroids(df, group_col='celltype')
    assert list(centroids.index) == ['CT-{}'.format(i) for i in range(1, 13)]


def test_celltype_centroids_median_differs_from_mean():
    df = pd.DataFrame({'X': [0.0, 0.0, 100.0], 'Y': [0.0, 0.0, 0.0],
                       'celltype': ['A', 'A', 'A']})
    mean = distances.celltype_centroids(df, group_col='celltype', method='mean')
    median = distances.celltype_centroids(df, group_col='celltype', method='median')
    assert np.isclose(mean.loc['A', 'X'], 100.0 / 3)
    assert np.isclose(median.loc['A', 'X'], 0.0)


def test_celltype_centroid_distances_is_a_distance_matrix(toy_spatial_adata):
    result = distances.celltype_centroid_distances(toy_spatial_adata, group_col='celltype')
    assert list(result.index) == list(result.columns)
    assert np.allclose(np.diag(result.values), 0.0)
    np.testing.assert_allclose(result.values, result.values.T)


def test_celltype_centroid_distances_known_geometry():
    df = pd.DataFrame({'X': [0.0, 0.0, 3.0, 3.0], 'Y': [0.0, 0.0, 4.0, 4.0],
                       'celltype': ['A', 'A', 'B', 'B']})
    result = distances.celltype_centroid_distances(df, group_col='celltype')
    assert np.isclose(result.loc['A', 'B'], 5.0)


@pytest.mark.parametrize('method', ['centroid', 'min', 'max', 'mean', 'median'])
def test_celltype_distances_methods(toy_spatial_adata, method):
    result = distances.celltype_distances(toy_spatial_adata, group_col='celltype',
                                          method=method)
    assert np.allclose(np.diag(result.values), 0.0)
    np.testing.assert_allclose(result.values, result.values.T)
    assert (result.values >= 0).all()


def test_celltype_distances_min_is_at_most_centroid():
    '''The closest two cells of two types cannot be further apart than their centroids.'''
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({'X': rng.random(n) * 10, 'Y': rng.random(n) * 10,
                       'celltype': ['A'] * (n // 2) + ['B'] * (n // 2)})
    closest = distances.celltype_distances(df, group_col='celltype', method='min')
    centroid = distances.celltype_distances(df, group_col='celltype', method='centroid')
    assert closest.loc['A', 'B'] <= centroid.loc['A', 'B'] + 1e-9


def test_celltype_distances_accepts_a_dataframe(toy_spatial_adata):
    coords = distances.get_spatial_coordinates(toy_spatial_adata)
    coords['celltype'] = np.asarray(toy_spatial_adata.obs['celltype'].values)
    from_df = distances.celltype_distances(coords, group_col='celltype')
    from_adata = distances.celltype_distances(toy_spatial_adata, group_col='celltype')
    np.testing.assert_allclose(from_df.values, from_adata.values)


def test_celltype_distances_custom_spatial_key(toy_spatial_adata):
    toy_spatial_adata.obsm['my_coords'] = toy_spatial_adata.obsm['spatial']
    custom = distances.celltype_distances(toy_spatial_adata, group_col='celltype',
                                          spatial_key='my_coords')
    default = distances.celltype_distances(toy_spatial_adata, group_col='celltype')
    np.testing.assert_allclose(custom.values, default.values)


def test_celltype_distances_manhattan_differs_from_euclidean():
    df = pd.DataFrame({'X': [0.0, 3.0], 'Y': [0.0, 4.0], 'celltype': ['A', 'B']})
    euclidean = distances.celltype_distances(df, group_col='celltype', distance='euclidean')
    manhattan = distances.celltype_distances(df, group_col='celltype', distance='manhattan')
    assert np.isclose(euclidean.loc['A', 'B'], 5.0)
    assert np.isclose(manhattan.loc['A', 'B'], 7.0)


def test_celltype_distances_rejects_unknown_group_col(toy_spatial_adata):
    with pytest.raises(KeyError):
        distances.celltype_distances(toy_spatial_adata, group_col='not_a_column')


def test_celltype_distances_rejects_bad_input():
    with pytest.raises(TypeError):
        distances.celltype_distances([1, 2, 3], group_col='celltype')
