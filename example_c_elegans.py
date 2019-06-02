import cell2cell as c2c

### SETUP
# Dictionaries to pass inputs
files = dict()
rnaseq_setup = dict()
ppi_setup = dict()
cutoff_setup = dict()
analysis_setup = dict()

# Files
files['rnaseq'] = './data/CElegans_RNASeqData_Cell.xlsx'
files['ppi'] = './data/CElegans_PPIs_RSPGM.xlsx'
files['go_annotations'] = './data/wb.gaf.gz'
files['go_terms'] = './data/go-basic.obo'
files['cutoffs'] = None  # Keep this None when a cutoff file is not provided (cutoff values will be computed instead)
files['output_folder'] = './outputs/'

# RNA-seq data setup
rnaseq_setup['gene_col'] = 'gene_id'  # Name of the column containing gene names
rnaseq_setup['drop_nangenes'] = True  # Drop genes with all values as nan in the original table
rnaseq_setup['log_transform'] = False  # To log transform RNA-seq data

# PPI network setup
ppi_setup['protein_cols'] = ['WormBase_ID_a', 'WormBase_ID_b']   # Name of columns for interacting proteins.
ppi_setup['score_col'] = 'RSPGM_Score'   # Name of column for interaction score or probability in the network - Not used in this example.

# Cutoff
cutoff_setup['type'] = 'percentile'
cutoff_setup['parameter'] = 0.8 # In this case is for percentile, representing to compute the 80-th percentile value.

# Analysis
analysis_setup['interaction_type'] = 'combined'
analysis_setup['subsampling_percentage'] = 0.8
analysis_setup['iterations'] = 1000
analysis_setup['go_descendants'] = True   # This is for including the descendant GO terms (hierarchically below) to filter PPIs
analysis_setup['clustering_algorithm'] = 'louvain'
analysis_setup['clustering_method'] = 'raw'
analysis_setup['verbose'] = False
analysis_setup['cpu_cores'] = 7 # To enable parallel computing

if __name__ == '__main__':
    import time
    start = time.time()

    # Load Data
    rnaseq_data = c2c.io.load_rnaseq(files['rnaseq'],
                                     rnaseq_setup['gene_col'],
                                     drop_nangenes=rnaseq_setup['drop_nangenes'],
                                     log_transformation=rnaseq_setup['log_transform'],
                                     format='auto')

    ppi_data = c2c.io.load_ppi(files['ppi'],
                               ppi_setup['protein_cols'],
                               score=ppi_setup['score_col'],
                               rnaseq_genes=list(rnaseq_data.index),
                               format='auto')

    go_annotations = c2c.io.load_go_annotations(files['go_annotations'])
    go_terms = c2c.io.load_go_terms(files['go_terms'])

    # Run Analysis
    if 'cutoffs' not in files.keys():
        cutoff_setup['file'] = None
    else:
        cutoff_setup['file'] = files['cutoffs']

    subsampling_space = c2c.analysis.run_analysis(rnaseq_data,
                                                  ppi_data,
                                                  go_annotations,
                                                  go_terms,
                                                  cutoff_setup,
                                                  analysis_setup)

    print("It took %.2f seconds" % (time.time() - start))

