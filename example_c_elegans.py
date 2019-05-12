import cell2cell as c2c

### DATA
# Files
RNAseq_file = './data/CElegans_RNASeqData_Cell.xlsx'
PPI_file = './data/CElegans_PPIs_RSPGM.xlsx'
GO_annotations_file = './data/wb.gaf.gz'
GO_terms_file = './data/go-basic.obo'

# PPI network setup
interactors = ['WormBase_ID_a', 'WormBase_ID_b']   # Columns name of interactors. They will be transforme to 'A' and 'B'.
score = 'RSPGM_Score'   # Columns name of interaction score or probability in the network - Not used in this example.

# RNA-seq data setup
gene_header = 'gene_id' # Name of the column containing gene names
cell_starting_index = 2
cutoff_percentile = 0.75

# Load Data
RNAseq_data = c2c.io.load_rnaseq(RNAseq_file, gene_header, format = 'excel')
PPI_data = c2c.io.load_ppi(PPI_file, interactors, score = score, rnaseq_genes=list(RNAseq_data.index), format = 'excel')
GO_data = c2c.io.load_go_annotations(GO_annotations_file)
GO_terms = c2c.io.load_go_terms(GO_terms_file)

### TESTS
cutoffs = RNAseq_data.mean(axis=1)
cutoffs = cutoffs.to_frame()
cutoffs.columns = ['value']

binary_rnaseq = c2c.preprocessing.get_binary_rnaseq(RNAseq_data, cutoffs)
binary_ppi = c2c.preprocessing.get_binary_ppi(PPI_data, binary_rnaseq, column='Germline')
