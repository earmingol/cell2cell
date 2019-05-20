import cell2cell as c2c

### SETUP
# Files
rnaseq_file = './data/CElegans_RNASeqData_Cell.xlsx'
ppi_file = './data/CElegans_PPIs_RSPGM.xlsx'
go_annotations_file = './data/wb.gaf.gz'
go_terms_file = './data/go-basic.obo'
cutoff_file = None  # Keep this None when a cutoff file is not provide (cutoff values will be computed here instead)

# PPI network setup
interactors = ['WormBase_ID_a', 'WormBase_ID_b']   # Columns name of interactors. They will be transforms to 'A' and 'B'.
score = 'RSPGM_Score'   # Columns name of interaction score or probability in the network - Not used in this example.

# RNA-seq data setup
gene_header = 'gene_id' # Name of the column containing gene names

# Cutoff
cutoff_type = 'percentile'
cutoff_paramater = 0.8 # In this case is for percentile

# Interaction to analyze
interaction_type = 'combined'

# Subsampling
subsampling_percentage = 0.8
iterations = 1000

# GO Terms to filter PPIs
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

if __name__ == '__main__':
    # Load Data
    rnaseq_data = c2c.io.load_rnaseq(rnaseq_file,
                                     gene_header,
                                     drop_nangenes=True,
                                     log_transformation=False,
                                     format='excel')

    ppi_data = c2c.io.load_ppi(ppi_file,
                               interactors,
                               score = score,
                               rnaseq_genes=list(rnaseq_data.index),
                               format='excel')

    go_annotations = c2c.io.load_go_annotations(go_annotations_file)
    go_terms = c2c.io.load_go_terms(go_terms_file)

    # Cutoffs dictionary
    gene_cutoffs = dict()
    gene_cutoffs['type'] = cutoff_type
    gene_cutoffs['parameter'] = cutoff_paramater
    gene_cutoffs['file'] = cutoff_file

    # Run analysis
    ppi_dict = c2c.preprocessing.ppis_from_goterms(ppi_data,
                                                   go_annotations,
                                                   go_terms,
                                                   contact_go_terms,
                                                   mediator_go_terms,
                                                   use_children=True,
                                                   verbose=True)

    import random

    cell_ids = list(rnaseq_data.columns)
    last_item = int(len(cell_ids) * subsampling_percentage)

    subsampling_space = []

    for i in range(iterations):
        print("PERFORMING SUBSAMPLING ITERATION NUMBER {} OUT OF {}".format(i, iterations))
        random.shuffle(cell_ids)
        included_cells = cell_ids[:last_item]

        interaction_space = c2c.interaction.space.InteractionSpace(rnaseq_data[included_cells],
                                                                   ppi_dict,
                                                                   interaction_type,
                                                                   gene_cutoffs,
                                                                   verbose=False)

        interaction_space.compute_pairwise_interactions(verbose=False)

        # clustering
        clusters = c2c.clustering.clustering_interactions(interaction_space.interaction_elements['cci_matrix'],
                                                          method='louvain')


        iteration_elements = (interaction_space.interaction_elements['cci_matrix'], clusters)
        subsampling_space.append(iteration_elements)


