# -*- coding: utf-8 -*-

from __future__ import absolute_import

# ==========================================================================================================
# DEFAULT - MODIFY THESE TO INCLUDE DIFFERENT SURFACE (CONTACT) AND EXTRACELLULAR (MEDIATOR) PROTEINS
# ==========================================================================================================
contact_go_terms = ['GO:0007155',  # Cell adhesion
                    'GO:0022608',  # Multicellular organism adhesion
                    'GO:0098740',  # Multiorganism cell adhesion
                    'GO:0098743',  # Cell aggregation
                    'GO:0030054',  # Cell-junction
                    'GO:0009986',  # Cell surface
                    'GO:0097610',  # Cell surface forrow
                    'GO:0005215',  # Transporter activity
                    ]

mediator_go_terms = ['GO:0005615',  # Extracellular space
                     #'GO:0012505',  # Endomembrane system - Some proteins are not secreted (they are intracellular).
                     'GO:0005576',  # Extracellular region
                     ]
# ==========================================================================================================


# ==========================================================================================================
# BADER LAB LIST - http://baderlab.org/CellCellInteractions
# ==========================================================================================================

receptors = ['GO:0043235', # receptor complex,
             'GO:0008305', # integrin complex,
             #'GO:0072657', # protein localized to membrane, - not in go terms anymore
             'GO:0043113', # receptor clustering
             #'GO:0004872', # receptor activity, - not in go terms anymore
             'GO:0009897', # external side of plasma membrane)
             ]

#ligands = ['GO:0005102'] # receptor binding - not in go terms anymore

extracellular = ['GO:0031012', # extracellular matrix
                 #'GO:0005578', # proteinacious extracellular matrix - not in go terms anymore
                 'GO:0005201', # extracellular matrix structural constituent
                 'GO:1990430', # extracellular matrix protein binding
                # 'GO:0035426', # extracellular matrix cell signalling - not in go terms anymore
                 ]
# ==========================================================================================================


def go_terms_from_keyword(go_terms, keyword):
    '''
    This function looks for GO terms containing a given keyword.

    Parameters
    ----------
    go_terms : networkx.Graph()
        NetworkX Graph containing GO terms data from .obo file. It could be loaded with goenrich package or using function
        cell2cell.io.load_go_terms(filename).

    keyword : str
        Keyword to be included in the names of retrieved GO terms.

    Returns
    -------
    go_filter : array-like
        List containing all GO terms that have keyword as part of their names.
    '''
    go_filter = []
    for go, node in go_terms.nodes.items():
        if keyword in node['name']:
            go_filter.append(go)
    return go_filter