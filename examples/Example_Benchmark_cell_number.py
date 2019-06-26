import cell2cell as c2c
import benchmarks

### SETUP
# Dictionaries to pass inputs
files = dict()
rnaseq_setup = dict()
ppi_setup = dict()
cutoff_setup = dict()
analysis_setup = dict()
benchmark_setup = dict()

# Files
files['rnaseq'] = '../data/CElegans_RNASeqData_Cell.xlsx'
files['ppi'] = '../data/CElegans_PPIs_RSPGM.xlsx'
files['go_annotations'] = '../data/wb.gaf.gz'
files['go_terms'] = '../data/go-basic.obo'
files['output_folder'] = '../outputs/'

# Benchmark setup
benchmark_setup['cell_number'] = 3000

# RNA-seq data setup
rnaseq_setup['gene_col'] = 'gene_id'  # Name of the column containing gene names
rnaseq_setup['drop_nangenes'] = True  # Drop genes with all values as nan in the original table
rnaseq_setup['log_transform'] = False  # To log transform RNA-seq data

# PPI network setup
ppi_setup['protein_cols'] = ['WormBase_ID_a', 'WormBase_ID_b']   # Name of columns for interacting proteins.
ppi_setup['score_col'] = 'RSPGM_Score'   # Name of column for interaction score or probability in the network - Not used in this example.

# Cutoff
cutoff_setup['type'] = 'local_percentile'
cutoff_setup['parameter'] = 0.75 # In this case is for percentile, representing to compute the 75-th percentile value.

# Analysis
analysis_setup['interaction_type'] = 'combined'
analysis_setup['subsampling_percentage'] = 0.8
analysis_setup['iterations'] = 5
analysis_setup['initial_seed'] = None # Turns on using a Seed for randomizing which cells are used in each iteration. Use None instead for not using a seed.
analysis_setup['goa_experimental_evidence'] = True # Consider experimental evidence to retrieve genes from GO terms
analysis_setup['go_descendants'] = True   # This is for including the descendant GO terms (hierarchically below) to filter PPIs
analysis_setup['clustering_algorithm'] = 'louvain'
analysis_setup['clustering_method'] = 'average'
analysis_setup['verbose'] = False
analysis_setup['cpu_cores'] = 7 # To enable parallel computing

if __name__ == '__main__':
    import time
    start = time.time()

    # Run Analysis
    subsampling_space, filtered_ppis = benchmarks.run_benchmark(files,
                                                                rnaseq_setup,
                                                                ppi_setup,
                                                                cutoff_setup,
                                                                analysis_setup,
                                                                benchmark_setup)

    print("It took %.2f seconds" % (time.time() - start))

