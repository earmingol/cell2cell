# -*- coding: utf-8 -*-

from scipy.signal import savgol_filter


def smooth_curve(values, window_length=None, polyorder=3, **kwargs):
    '''Apply a Savitzky-Golay filter to an array to smooth the curve.

    Parameters
    ----------
    values : array-like
        An array or list of values.

    window_length : int, default=None
        Size of the window of values to use too smooth the curve.

    polyorder : int, default=3
        The order of the polynomial used to fit the samples.

    **kwargs : dict
        Extra arguments for the scipy.signal.savgol_filter function.

    Returns
    -------
    smooth_values : array-like
        An array or list of values representing the smooth curvee.
    '''
    size = len(values)
    if window_length is None:
        window_length = int(size / min([2, size]))
    if window_length % 2 == 0:
        window_length += 1
    assert(polyorder < window_length), "polyorder must be less than window_length."
    smooth_values = savgol_filter(values, window_length, polyorder, **kwargs)
    return smooth_values