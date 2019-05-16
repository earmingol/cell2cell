import cell2cell as c2c

### SETUP
# Files
rnaseq_file = './data/CElegans_RNASeqData_Cell.xlsx'
ppi_file = './data/CElegans_PPIs_RSPGM.xlsx'
go_annotations_file = './data/wb.gaf.gz'
go_terms_file = './data/go-basic.obo'
cutoff_file = None

# PPI network setup
interactors = ['WormBase_ID_a', 'WormBase_ID_b']   # Columns name of interactors. They will be transforme to 'A' and 'B'.
score = 'RSPGM_Score'   # Columns name of interaction score or probability in the network - Not used in this example.

# RNA-seq data setup
gene_header = 'gene_id' # Name of the column containing gene names

# Cutoffs dictionary
gene_cutoffs = dict()
gene_cutoffs['type'] = 'percentile'
gene_cutoffs['parameter'] = 0.8 # In this case is for percentyle
gene_cutoffs['file'] = cutoff_file

# Interaction to analyze
interaction_type = 'combined'

# Subsampling percentage
subsampling_percentage = 0.8


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
                     #'GO:0012505',  # Endomembrane system | Some proteins could be not secreted.
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

    go_data = c2c.io.load_go_annotations(go_annotations_file)
    go_terms = c2c.io.load_go_terms(go_terms_file)


    # Run analysis

    #binary_rnaseq = c2c.preprocessing.get_binary_rnaseq(rnaseq_data, gene_cutoffs['dataframe'])
    #binary_ppi = c2c.preprocessing.get_binary_ppi(ppi_data, binary_rnaseq, column='Germline')

    contact_proteins = c2c.preprocessing.get_genes_from_parent_go_terms(go_data,
                                                                        go_terms,
                                                                        contact_go_terms,
                                                                        go_header = 'GO',
                                                                        gene_header = 'Gene',
                                                                        verbose = False)

    mediator_proteins = c2c.preprocessing.get_genes_from_parent_go_terms(go_data,
                                                                         go_terms,
                                                                         mediator_go_terms,
                                                                         go_header = 'GO',
                                                                         gene_header = 'Gene',
                                                                         verbose = False)

    ppi_dict = dict()
    for interaction in ['contacts', 'mediated', 'combined']:
        ppi_dict[interaction] = c2c.preprocessing.filter_ppi_network(ppi_data,
                                                                     contact_proteins,
                                                                     mediator_proteins,
                                                                     interaction_type=interaction)
        #ppi_dict[interaction] = ppi_data

    interaction_space = c2c.interaction.space.InteractionSpace(rnaseq_data,
                                                               ppi_dict,
                                                               interaction_type,
                                                               gene_cutoffs,
                                                               verbose=True)

    interaction_space.compute_pairwise_interactions()
