# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import pandas as pd
import random

from cell2cell.core import InteractionSpace
from cell2cell.utils import parallel_computing
from contextlib import closing
from multiprocessing import Pool


class SpatialCCI:

    def __init__(self, rnaseq_data, ppi_data, gene_cutoffs, spatial_dict, score_type='binary', score_metric='bray_curtis',
                 n_jobs=1, verbose=True):
        pass
        # TODO IMPLEMENT SPATIAL CCI