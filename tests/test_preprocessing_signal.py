# -*- coding: utf-8 -*-

'''Tests for cell2cell.preprocessing.signal'''

import numpy as np
import pytest

from cell2cell.preprocessing import signal


def test_smooth_curve_preserves_length():
    values = [5.0, 3.0, 4.0, 2.0, 3.5, 1.0, 2.0, 0.5, 1.5, 0.2]
    smoothed = signal.smooth_curve(values)
    assert len(smoothed) == len(values)


def test_smooth_curve_reduces_the_variation():
    noisy = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0]
    smoothed = signal.smooth_curve(noisy)
    assert np.std(smoothed) < np.std(noisy)


def test_smooth_curve_with_an_explicit_window():
    values = list(np.linspace(1.0, 0.1, 20))
    smoothed = signal.smooth_curve(values, window_length=5, polyorder=2)
    assert len(smoothed) == len(values)
    assert np.isfinite(smoothed).all()


def test_smooth_curve_on_a_straight_line_is_almost_unchanged():
    values = list(np.linspace(0.0, 1.0, 15))
    smoothed = signal.smooth_curve(values, window_length=5, polyorder=2)
    assert np.allclose(smoothed, values, atol=1e-8)


def test_smooth_curve_accepts_a_numpy_array():
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0])
    smoothed = signal.smooth_curve(values)
    assert np.isfinite(smoothed).all()


def test_smooth_curve_is_deterministic():
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]
    assert np.allclose(signal.smooth_curve(values), signal.smooth_curve(values))
