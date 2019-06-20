import cell2cell as c2c

### SETUP
# Dictionaries to pass inputs
files = dict()
rnaseq_setup = dict()
ppi_setup = dict()
cutoff_setup = dict()
analysis_setup = dict()

# Files
files['rnaseq'] = '../data/CElegans_RNASeqData_Cell.xlsx'
files['ppi'] = '../data/CElegans_PPIs_RSPGM.xlsx'
files['go_annotations'] = '../data/wb.gaf.gz'
files['go_terms'] = '../data/go-basic.obo'
files['output_folder'] = '../outputs/'

# RNA-seq data setup
rnaseq_setup['gene_col'] = 'gene_id'  # Name of the column containing gene names
rnaseq_setup['drop_nangenes'] = True  # Drop genes with all values as nan in the original table
rnaseq_setup['log_transform'] = False  # To log transform RNA-seq data

# PPI network setup
ppi_setup['protein_cols'] = ['WormBase_ID_a', 'WormBase_ID_b']   # Name of columns for interacting proteins.
ppi_setup['score_col'] = 'RSPGM_Score'   # Name of column for interaction score or probability in the network - Not used in this example.

# Cutoff
cutoff_setup['type'] = 'local_percentile'
cutoff_setup['parameter'] = 0.8 # In this case is for percentile, representing to compute the 80-th percentile value.

# Analysis
analysis_setup['interaction_type'] = 'combined'
analysis_setup['subsampling_percentage'] = 0.8
analysis_setup['iterations'] = 100
analysis_setup['initial_seed'] = None # Turns on using a Seed for randomizing which cells are used in each iteration. Use None instead for not using a seed.
analysis_setup['goa_experimental_evidence'] = False # True for considering experimental evidence to retrieve genes from GO terms
analysis_setup['go_descendants'] = True   # This is for including the descendant GO terms (hierarchically below) to filter PPIs
analysis_setup['clustering_algorithm'] = 'louvain'
analysis_setup['clustering_method'] = 'raw'
analysis_setup['verbose'] = False
analysis_setup['cpu_cores'] = 7 # To enable parallel computing

if __name__ == '__main__':
    import time
    start = time.time()

    # Load Data
    rnaseq_data = c2c.io.load_rnaseq(rnaseq_file=files['rnaseq'],
                                     gene_column=rnaseq_setup['gene_col'],
                                     drop_nangenes=rnaseq_setup['drop_nangenes'],
                                     log_transformation=rnaseq_setup['log_transform'],
                                     format='auto')

    ppi_data = c2c.io.load_ppi(ppi_file=files['ppi'],
                               interaction_columns=ppi_setup['protein_cols'],
                               score=ppi_setup['score_col'],
                               rnaseq_genes=list(rnaseq_data.index),
                               format='auto')

    go_annotations = c2c.io.load_go_annotations(files['go_annotations'],
                                                experimental_evidence=analysis_setup['goa_experimental_evidence'])
    go_terms = c2c.io.load_go_terms(files['go_terms'])

    # Run analysis
    ppi_dict = c2c.preprocessing.ppis_from_goterms(ppi_data=ppi_data,
                                                   go_annotations=go_annotations,
                                                   go_terms=go_terms,
                                                   contact_go_terms=c2c.core.contact_go_terms,
                                                   mediator_go_terms=c2c.core.mediator_go_terms,
                                                   use_children=analysis_setup['go_descendants'],
                                                   verbose=False)

    print("Running analysis with a subsampling percentage of {}% and {} iterations".format(
        int(100 * analysis_setup['subsampling_percentage']), analysis_setup['iterations'])
    )

    subsampling_space = c2c.analysis.SubsamplingSpace(rnaseq_data=rnaseq_data,
                                                      ppi_dict=ppi_dict,
                                                      interaction_type=analysis_setup['interaction_type'],
                                                      gene_cutoffs=cutoff_setup,
                                                      subsampling_percentage=analysis_setup['subsampling_percentage'],
                                                      iterations=analysis_setup['iterations'],
                                                      n_jobs=analysis_setup['cpu_cores'],
                                                      verbose=analysis_setup['verbose'])

    print("Performing clustering with {} algorithm and {} method".format(analysis_setup['clustering_algorithm'],
                                                                         analysis_setup['clustering_method']))

    clustering = c2c.clustering.clustering_interactions(interaction_elements=subsampling_space.subsampled_interactions,
                                                        algorithm=analysis_setup['clustering_algorithm'],
                                                        method=analysis_setup['clustering_method'],
                                                        verbose=False)

    subsampling_space.clustering = clustering

    # Tryng clustering methods
    hier_community = c2c.clustering.hierarchical_community(subsampling_space.clustering['final_cci_matrix'], algorithm='community_fastgreedy')

    print(hier_community)
    print("It took %.2f seconds" % (time.time() - start))

