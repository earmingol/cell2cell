# -*- coding: utf-8 -*-

'''Tests for cell2cell.clustering'''

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import squareform

from cell2cell import clustering


def test_compute_distance_shape_and_symmetry(toy_rnaseq):
    result = clustering.compute_distance(toy_rnaseq, axis=0, metric='euclidean')
    n = toy_rnaseq.shape[0]
    assert result.shape == (n, n)
    assert np.allclose(result, result.T)
    assert np.allclose(np.diag(result), 0.0)


def test_compute_distance_along_the_other_axis(toy_rnaseq):
    result = clustering.compute_distance(toy_rnaseq, axis=1, metric='euclidean')
    assert result.shape == (toy_rnaseq.shape[1], toy_rnaseq.shape[1])


@pytest.mark.parametrize('metric', ['euclidean', 'cityblock', 'cosine'])
def test_compute_distance_metrics(toy_rnaseq, metric):
    result = clustering.compute_distance(toy_rnaseq, axis=0, metric=metric)
    assert (result >= -1e-12).all()


def test_compute_distance_matches_scipy(toy_rnaseq):
    from scipy.spatial.distance import pdist
    result = clustering.compute_distance(toy_rnaseq, axis=0, metric='euclidean')
    expected = squareform(pdist(toy_rnaseq.values, metric='euclidean'))
    assert np.allclose(result, expected)


def test_compute_linkage_shape(toy_distance):
    linkage = clustering.compute_linkage(toy_distance, method='ward')
    assert linkage.shape == (toy_distance.shape[0] - 1, 4)


@pytest.mark.parametrize('method', ['ward', 'average', 'complete', 'single'])
def test_compute_linkage_methods(toy_distance, method):
    linkage = clustering.compute_linkage(toy_distance, method=method)
    assert linkage.shape[0] == toy_distance.shape[0] - 1
    # Merge distances must be non-decreasing for these methods
    assert np.all(np.diff(linkage[:, 2]) >= -1e-9)


def test_compute_linkage_is_deterministic(toy_distance):
    first = clustering.compute_linkage(toy_distance, method='ward')
    second = clustering.compute_linkage(toy_distance, method='ward')
    assert np.allclose(first, second)


def test_compute_linkage_accepts_a_read_only_frame(read_only_frame, toy_distance):
    '''compute_linkage zeroes the diagonal in place, which used to be done on the array
    behind `DataFrame.values`. That array is read-only under the copy-on-write of
    pandas >= 3.0. Only the ndarray branch of the function used to copy it.'''
    frame = read_only_frame(toy_distance, labels=list(toy_distance.index))
    linkage = clustering.compute_linkage(frame, method='ward')
    assert np.allclose(linkage, clustering.compute_linkage(toy_distance, method='ward'))


def test_compute_linkage_accepts_a_read_only_array(toy_distance):
    array = np.array(toy_distance, dtype=float)
    array.setflags(write=False)
    linkage = clustering.compute_linkage(array, method='ward')
    assert np.allclose(linkage, clustering.compute_linkage(toy_distance, method='ward'))


def test_compute_linkage_does_not_modify_its_input(toy_distance):
    similarity = 1 - toy_distance / toy_distance.values.max()
    before = similarity.copy()
    clustering.compute_linkage(similarity, method='average')
    pd.testing.assert_frame_equal(similarity, before)


def test_get_clusters_from_linkage_maxclust(toy_distance):
    linkage = clustering.compute_linkage(toy_distance, method='ward')
    clusters = clustering.get_clusters_from_linkage(linkage, threshold=2,
                                                   criterion='maxclust',
                                                   labels=list(toy_distance.index))
    assert len(clusters) == 2
    members = [m for group in clusters.values() for m in group]
    assert sorted(members) == sorted(list(toy_distance.index))


def test_get_clusters_from_linkage_without_labels(toy_distance):
    linkage = clustering.compute_linkage(toy_distance, method='ward')
    clusters = clustering.get_clusters_from_linkage(linkage, threshold=3,
                                                   criterion='maxclust')
    assert len(clusters) == 3
    total = sum(len(group) for group in clusters.values())
    assert total == toy_distance.shape[0]


def test_get_clusters_from_linkage_every_element_appears_once(toy_distance):
    linkage = clustering.compute_linkage(toy_distance, method='average')
    clusters = clustering.get_clusters_from_linkage(linkage, threshold=2,
                                                   criterion='maxclust',
                                                   labels=list(toy_distance.index))
    members = [m for group in clusters.values() for m in group]
    assert len(members) == len(set(members))


def test_clustering_pipeline_end_to_end(toy_rnaseq):
    distance = clustering.compute_distance(toy_rnaseq, axis=1)
    frame = pd.DataFrame(distance, index=toy_rnaseq.columns, columns=toy_rnaseq.columns)
    linkage = clustering.compute_linkage(frame, method='ward')
    clusters = clustering.get_clusters_from_linkage(linkage, threshold=2,
                                                   criterion='maxclust',
                                                   labels=list(toy_rnaseq.columns))
    assert len(clusters) == 2
