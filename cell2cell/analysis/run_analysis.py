# -*- coding: utf-8 -*-

from __future__ import absolute_import

import cell2cell as c2c

#==========================================================================================================
# MODIFY THESE TO INCLUDE DIFFERENT SURFACE (CONTACT) AND EXTRACELLULAR (MEDIATOR) PROTEINS
#==========================================================================================================
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
                     #'GO:0012505',  # Endomembrane system | Some proteins are not secreted.
                     'GO:0005576',  # Extracellular region
                    ]
#==========================================================================================================

def run_analysis(rnaseq_data, ppi_data, go_annotations, go_terms, cutoff_setup, analysis_setup):
    # Run analysis
    ppi_dict = c2c.preprocessing.ppis_from_goterms(ppi_data,
                                                   go_annotations,
                                                   go_terms,
                                                   contact_go_terms,
                                                   mediator_go_terms,
                                                   use_children=analysis_setup['go_descendants'],
                                                   verbose=False)

    print("Running analysis with a subsampling percentage of {}% and {} iterations".format(
          int(100*analysis_setup['subsampling_percentage']), analysis_setup['iterations'])
          )
    subsampled_interactions = c2c.analysis.SubsamplingSpace(rnaseq_data=rnaseq_data,
                                                            ppi_dict=ppi_dict,
                                                            interaction_type=analysis_setup['interaction_type'],
                                                            gene_cutoffs=cutoff_setup,
                                                            subsampling_percentage=analysis_setup['subsampling_percentage'],
                                                            iterations=analysis_setup['iterations'],
                                                            n_jobs=analysis_setup['cpu_cores'],
                                                            verbose=analysis_setup['verbose'])

    return subsampled_interactions