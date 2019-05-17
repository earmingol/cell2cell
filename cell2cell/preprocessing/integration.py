# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pandas as pd

from cell2cell.preprocessing.ppi import filter_ppi_network


### Integrate all datasets
def get_all_ppis(ppi_data, contact_proteins, mediator_proteins = None, interaction_columns = ['A', 'B']):

    all_ppis = dict()
    all_ppis['contacts'] = filter_ppi_network(ppi_data,
                                              contact_proteins,
                                              mediator_proteins=mediator_proteins,
                                              interaction_type='contacts',
                                              interaction_columns=interaction_columns)
    if mediator_proteins is not None:
        all_ppis['mediated'] = filter_ppi_network(ppi_data,
                                                  contact_proteins,
                                                  mediator_proteins=mediator_proteins,
                                                  interaction_type='mediated',
                                                  interaction_columns=interaction_columns)

        all_ppis['combined'] = filter_ppi_network(ppi_data,
                                                  contact_proteins,
                                                  mediator_proteins=mediator_proteins,
                                                  interaction_type='combined',
                                                  interaction_columns=interaction_columns)
    # else:
    #     columns = interaction_columns + ['score']
    #     all_ppis['mediated'] = pd.DataFrame(columns=columns)
    #     all_ppis['combined'] = pd.DataFrame(columns=columns)
    # This is omitted because other functions use all_ppis.values() instead of checking all types of interactions.
    return all_ppis


