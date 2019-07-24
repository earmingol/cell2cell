# -*- coding: utf-8 -*-

from __future__ import absolute_import

from cell2cell.clustering import cluster_interactions
from cell2cell.core import subsampling
from cell2cell.datasets import heuristic_data
from cell2cell.io import read_data
from cell2cell.preprocessing import integrate_data


def run_heuristic_pipeline(files, rnaseq_setup, ppi_setup, cutoff_setup, analysis_setup,
                           contact_go_terms = None, mediator_go_terms = None):
    '''
    This function performs the analysis with the DEFAULT list of GO terms to filter the PPI network.
    '''

    rnaseq_data = read_data.load_rnaseq(rnaseq_file=files['rnaseq'],
                                        gene_column=rnaseq_setup['gene_col'],
                                        drop_nangenes=rnaseq_setup['drop_nangenes'],
                                        log_transformation=rnaseq_setup['log_transform'],
                                        format='auto')

    ppi_data = read_data.load_ppi(ppi_file=files['ppi'],
                                  interaction_columns=ppi_setup['protein_cols'],
                                  score=ppi_setup['score_col'],
                                  rnaseq_genes=list(rnaseq_data.index),
                                  format='auto')

    go_annotations = read_data.load_go_annotations(goa_file=files['go_annotations'],
                                                   experimental_evidence=analysis_setup['goa_experimental_evidence'])

    go_terms = read_data.load_go_terms(go_terms_file=files['go_terms'])

    default_heuristic_go = heuristic_data.HeuristicGO()

    if contact_go_terms is None:
        contact_go_terms = default_heuristic_go.contact_go_terms

    if mediator_go_terms is None:
        mediator_go_terms = default_heuristic_go.mediator_go_terms

    # Run analysis
    ppi_dict = integrate_data.get_ppi_dict_from_go_terms(ppi_data=ppi_data,
                                                         go_annotations=go_annotations,
                                                         go_terms=go_terms,
                                                         contact_go_terms=contact_go_terms,
                                                         mediator_go_terms=mediator_go_terms,
                                                         use_children=analysis_setup['go_descendants'],
                                                         verbose=False)

    print("Running analysis with a sub-sampling percentage of {}% and {} iterations".format(
          int(100*analysis_setup['subsampling_percentage']), analysis_setup['iterations'])
          )

    if 'initial_seed' not in analysis_setup.keys():
        initial_seed = None
    else:
        initial_seed = analysis_setup['initial_seed']

    subsampling_space = subsampling.SubsamplingSpace(rnaseq_data=rnaseq_data,
                                                     ppi_dict=ppi_dict,
                                                     interaction_type=analysis_setup['interaction_type'],
                                                     gene_cutoffs=cutoff_setup,
                                                     score_type=analysis_setup['function_type'],
                                                     subsampling_percentage=analysis_setup['subsampling_percentage'],
                                                     iterations=analysis_setup['iterations'],
                                                     initial_seed=initial_seed,
                                                     n_jobs=analysis_setup['cpu_cores'],
                                                     verbose=analysis_setup['verbose'])


    print("Performing clustering with {} algorithm and {} method".format(analysis_setup['clustering_algorithm'],
                                                                         analysis_setup['clustering_method']))

    clustering = cluster_interactions.graph_clustering(subsampled_interactions=subsampling_space.subsampled_interactions,
                                                       algorithm=analysis_setup['clustering_algorithm'],
                                                       method=analysis_setup['clustering_method'],
                                                       seed=None,
                                                       n_jobs=analysis_setup['cpu_cores'],
                                                       verbose=analysis_setup['verbose'])

    subsampling_space.graph_clustering = clustering

    #print(clustering['clusters'].sort_values(by='Cluster'))
    return subsampling_space, ppi_dict