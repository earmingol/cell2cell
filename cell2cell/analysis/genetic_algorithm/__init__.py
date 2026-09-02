# -*- coding: utf-8 -*-

'''Selection of ligand-receptor pairs with a genetic algorithm.

Reimplements, as part of the package, the analysis in
https://github.com/LewisLabUCSD/Celegans-cell2cell, which searches for the subset
of ligand-receptor pairs whose cell-cell interaction scores best reproduce a
reference distance between cells.

Reference
---------
Armingol E, Ghaddar A, Joshi CJ, Baghdassarian H, Shamie I, Chan J, et al. (2022)
Inferring a spatial code of cell-cell interactions across a whole animal body.
PLOS Computational Biology 18(11): e1010715.
https://doi.org/10.1371/journal.pcbi.1010715
'''

from __future__ import absolute_import

from cell2cell.core.prepared_scorer import (PreparedCCIScorer, LINEAR_CCI_SCORES,
                                            UNBOUNDED_CCI_SCORES)
from cell2cell.preprocessing.ppi import bidirectional_index

from cell2cell.analysis.genetic_algorithm.base import (CombinedObjective, COMBINERS)
from cell2cell.analysis.genetic_algorithm.objectives import (CorrelationObjective,
                                                             correlation_fitness,
                                                             _as_symmetric)
from cell2cell.analysis.genetic_algorithm.consensus import (lr_selection_frequency,
                                                            lr_cooccurrence,
                                                            consensus_from_cooccurrence,
                                                            consensus_from_frequency)
from cell2cell.analysis.genetic_algorithm.search import (optimize_lr_pairs, _optimize_once,
                                                         _check_if_pygad, _bidirectional_index)

# `_correlation` was the private name before the objective became pluggable
_correlation = correlation_fitness
