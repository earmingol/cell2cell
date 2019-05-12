# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd

from cell2cell.preprocessing.ppi import filter_ppi_network
from cell2cell.preprocessing.binary import get_binary_rnaseq, get_binary_ppi


### Integrate all datasets
def get_all_ppis(ppi_data, contact_proteins, mediator_proteins = None, interaction_columns = ['A', 'B']):

    all_ppis = dict()
    all_ppis['contacts'] = filter_ppi_network(ppi_data, contact_proteins, mediator_proteins, 'contacts', interaction_columns)
    if mediator_proteins is not None:
        all_ppis['mediated'] = filter_ppi_network(ppi_data, contact_proteins, mediator_proteins, 'mediated', interaction_columns )
        all_ppis['combined'] = filter_ppi_network(ppi_data, contact_proteins, mediator_proteins, 'combined', interaction_columns)
    # else:
    #     columns = interaction_columns + ['score']
    #     all_ppis['mediated'] = pd.DataFrame(columns=columns)
    #     all_ppis['combined'] = pd.DataFrame(columns=columns)
    # This is omitted because other functions use all_ppis.values() instead of checking all types of interactions.
    return all_ppis


def transform_to_binary_cell(cell, all_ppis, cutoffs, interaction_type = 'contacts'):
    ppi_data = all_ppis[interaction_type].copy()
    binary_rnaseq = get_binary_rnaseq(cell.rnaseq_data, cutoffs)
    cell.binary_ppi = get_binary_ppi(binary_rnaseq, ppi_data)


