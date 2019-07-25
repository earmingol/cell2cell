# -*- coding: utf-8 -*-

from __future__ import absolute_import

import time


def timeit(func, *args, **kwargs):
    '''
    This function measures the running time of a given function.
    Borrowed from George Armstrong's Github repo (https://github.com/gwarmstrong).
    '''
    t0 = time.time()
    output = func(*args, **kwargs)
    t1 = time.time()
    tot_time = t1-t0
    data = {'time': tot_time, 'results': output}
    return data