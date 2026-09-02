# -*- coding: utf-8 -*-

'''Tests for cell2cell.external.umap'''

import numpy as np

import cell2cell as c2c


def test_run_umap_shape(toy_rnaseq):
    result = c2c.external.run_umap(toy_rnaseq, axis=1, n_neighbors=3, random_state=0)
    assert result.shape[0] == toy_rnaseq.shape[1]
    assert result.shape[1] == 2


def test_run_umap_is_reproducible_with_a_seed(toy_rnaseq):
    first = c2c.external.run_umap(toy_rnaseq, axis=1, n_neighbors=3, random_state=7)
    second = c2c.external.run_umap(toy_rnaseq, axis=1, n_neighbors=3, random_state=7)
    assert np.allclose(first.values, second.values)


def test_run_umap_on_the_other_axis(toy_rnaseq):
    result = c2c.external.run_umap(toy_rnaseq, axis=0, n_neighbors=3, random_state=0)
    assert result.shape[0] == toy_rnaseq.shape[0]
