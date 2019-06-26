import cell2cell as c2c

### SETUP
# Dictionaries to pass inputs
files = dict()

# Files
#files['go_annotations'] = 'http://geneontology.org/gene-associations/goa_human.gaf.gz' # Human
#files['go_annotations'] = 'http://current.geneontology.org/annotations/mgi.gaf.gz' # Mouse
files['go_annotations'] = '../data/wb.gaf.gz' # C. elegans
files['go_terms'] = '../data/go-basic.obo'

if __name__ == '__main__':
    go_annotations = c2c.io.load_go_annotations(files['go_annotations'],
                                                experimental_evidence=True)
    go_terms = c2c.io.load_go_terms(files['go_terms'])

    go_filter = c2c.core.contact_go_terms + c2c.core.mediator_go_terms

    genes = c2c.preprocessing.get_genes_from_parent_go_terms(go_annotations,
                                                             go_terms,
                                                             go_filter,
                                                             gene_header='Name')

    print("{} genes were retrieved from GO terms".format(len(genes)))
    print(*genes, sep="\n")