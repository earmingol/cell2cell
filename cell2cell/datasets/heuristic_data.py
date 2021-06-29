# -*- coding: utf-8 -*-

from __future__ import absolute_import


class HeuristicGOTerms:
    '''GO terms for contact and secreted proteins.

    Attributes
    ----------
    contact_go_terms : list
        List of GO terms associated with proteins that
        participate in contact interactions (usually
        on the surface of cells).

    mediator_go_terms : list
        List of GO terms associated with secreted
        proteins that mediate intercellular interactions
        or communication.
    '''
    def __init__(self):
        self.contact_go_terms = ['GO:0007155',  # Cell adhesion
                                 'GO:0022608',  # Multicellular organism adhesion
                                 'GO:0098740',  # Multiorganism cell adhesion
                                 'GO:0098743',  # Cell aggregation
                                 'GO:0030054',  # Cell-junction #
                                 'GO:0009986',  # Cell surface #
                                 'GO:0097610',  # Cell surface forrow
                                 'GO:0007160',  # Cell-matrix adhesion
                                 'GO:0043235',  # Receptor complex,
                                 'GO:0008305',  # Integrin complex,
                                 'GO:0043113',  # Receptor clustering
                                 'GO:0009897',  # External side of plasma membrane #
                                 'GO:0038023',  # Signaling receptor activity #
                                 ]

        self.mediator_go_terms = ['GO:0005615',  # Extracellular space
                                  'GO:0005576',  # Extracellular region
                                  'GO:0031012',  # Extracellular matrix
                                  'GO:0005201',  # Extracellular matrix structural constituent
                                  'GO:1990430',  # Extracellular matrix protein binding
                                  'GO:0048018',  # Receptor ligand activity #
                                  ]