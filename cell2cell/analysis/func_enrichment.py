import goenrich
import mygene
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def initialize_go(background_id, obo_path = '/home/hsher/GO/go-basic.obo', gaf_path = '/home/hsher/GO/goa_human.gaf',gene2go_path = '/home/hsher/GO/gene2go'):
    '''
    a function to intiailize GO enrichement analysis with background gene sets
    background_id: ensembl gene id list for background gene set
    obo_path: go-basic.obo, download it here http://geneontology.org/docs/download-ontology/
    gaf_path: goa_human.gaf file path
    gene2go_path: gene2go file path
    
    return: O
    '''
    
    # build the ontology
    O = goenrich.obo.ontology(obo_path)

    # use all entrez geneid associations form gene2go as background
    annot = goenrich.read.goa(gaf_path)
    
    # genes to GO term mapping
    gene2go = goenrich.read.gene2go(gene2go_path)
    

    # subsetting all annot with only RNA-seq secreted proteins
    subset = gene2go.loc[gene2go['GeneID'].isin(background_id)]
    print('size of subset gene2go of gene2go is', subset.shape)
    values = {k: set(v) for k,v in subset.groupby('GO_ID')['GeneID']}
    
    # propagate the background through the ontology
    background_attribute = 'gene2go'
    goenrich.enrich.propagate(O, values, background_attribute)
    
    # output to file
    

    return(O)

def go_enrichment(O, query_set):
    '''
    run GO term enrichment for query set
    O: goenrich object from `initialize_go`
    query_set: ensembl gene id of list interested.
    
    return: dataframe with enriched go terms
    '''
    background_attribute = 'gene2go'
    
    # for additional export to graphviz just specify the gvfile argument
    # the show argument keeps the graph reasonably small
    df = goenrich.enrich.analyze(O, query_set, background_attribute, gvfile='test.dot')
    
    return(df.loc[df['rejected']==1].sort_values(by = 'p'))

def build_id_lookup(rnaseq_data, rnaseq_col = None, use_index = False, scopes = 'symbol', species = 9606):
    '''
    helper function to convert whatever id to ensembl id using mygene
    rnaseq_data: generate all id lookup at once
    rnaseq_col: column that contains all the gene symbol. Don't need to specify if use_index = True.
    use_index: if rnaseq_data stores gene name in index
    
    return: dataframe with all found list
    '''
    # initialize mygene
    mg = mygene.MyGeneInfo()
    
    if use_index:
        genes = rnaseq_data.index
    else:
        genes = rnaseq_data[rnaseq_col]
    
    lookup_table = mg.querymany(genes, scopes=scopes, fields='entrezgene', as_dataframe=True, species=species)
    
    # discard not found items
    lookup_table = lookup_table.loc[lookup_table['entrezgene'].notnull()]
    
    # convert id to integer for later
    lookup_table['entrezgene'] = lookup_table['entrezgene'].astype(int)
    
    return(lookup_table)

def symbol_to_entrezgene(gene_set, lookup_table):
    '''
    helper function to convert any gene set for ensembl id
    require prebuilt lookup table by calling `build_id_lookup`
    '''
    return(lookup_table.loc[lookup_table.index.isin(gene_set), 'entrezgene'].dropna().tolist()) 

def interacting_protein(interaction_matrix, pair):
    '''
    return interacting protein between cell pairs
    mind the directionality of proteins
    '''
    
    # pick interaction protein for certain cell pairs
    interacting_pair = interaction_matrix.loc[interaction_matrix[pair] >0].index
    
      
    # only on one side
    first_cell_protein = [i[0] for i in interacting_pair]
    second_cell_protein = [i[1] for i in interacting_pair]
    
    return(set(first_cell_protein), set(second_cell_protein))

def bidir_interacting_protein(interaction_matrix, cell_one, cell_two):
    '''
    return 1;9 9;1
    '''
    pair = str(cell_one)+';'+str(cell_two)
    first_cell_protein_1, second_cell_protein_1 = interacting_protein(interaction_matrix, pair)
    
    pair = str(cell_two)+';'+str(cell_one)
    second_cell_protein_2, first_cell_protein_2 = interacting_protein(interaction_matrix, pair)
    
    return(first_cell_protein_1.union(first_cell_protein_2), second_cell_protein_1.union(second_cell_protein_2))

def all_pairs_interaction(interaction_matrix):
    '''
    return ndarray containing all interating proteins between all cell types (ensembl id to connect reactome analysis)
    '''
    # how many cells
    no_cell_type = len(set([pair.split(';')[1] for pair in interaction_matrix.columns]))
    all_protein = np.ndarray(shape = (no_cell_type, no_cell_type, 2), dtype = object) # axis 1 and 2: cell type; 1 is the contributing; 2 is its partner, axis 3: R/L
    
    for pair in interaction_matrix.columns:
        # index
        c1_id = int(pair.split(';')[0]) # R
        c2_id = int(pair.split(';')[1]) # L
        
        # for each pair, extract interating rotein for first cell `f` and second cell `s`
        first_cell_protein, second_cell_protein = interacting_protein(interaction_matrix, pair)
        
              
        # save into ndarray
        all_protein[c1_id, c2_id, 0] = first_cell_protein # itself, parter, R/L
        all_protein[c2_id, c1_id, 1] = second_cell_protein
    return(all_protein)
        
def interaction_background(interaction_matrix):
    '''
    return
    all expressed interating protein 
    to serve as background for go term enrichment of cell-cell itxn
    '''
    proteins = set([p[0] for p in interaction_matrix.index] + [p[1] for p in interaction_matrix.index])
    R_protein = set([p[0] for p in interaction_matrix.index])
    L_protein = set([p[1] for p in interaction_matrix.index])
    return(proteins, R_protein, L_protein)

def enriched_interaction(interaction_matrix, lookup_table, cell_one, cell_two):
    '''
    a function to run enrichment analysis on proteins interacting between cell pairs
    interaction_matrix:  returned by c2c.analysis.get_ppi_score_for_cell_pairs
    lookup_table: gene symbol table precalculated
    '''
    
    # target set: call for bidiretionality and 
    first_cell_protein, second_cell_protein = bidir_interacting_protein(interaction_matrix, cell_one, cell_two)
    first_cell = symbol_to_entrezgene(first_cell_protein, lookup_table)
    second_cell = symbol_to_entrezgene(second_cell_protein, lookup_table)
    
    # run go enrichment,  background gene set: all expressed interating protein specified in interaction matrix
    all_exp_itxn, R_protein, L_protein = interaction_background(interaction_matrix)
    all_exp_itxn = symbol_to_entrezgene(all_exp_itxn, lookup_table)
    O = initialize_go(all_exp_itxn)
    first_enrichment = go_enrichment(O, first_cell)
    
    # output to file
    # call to graphviz
    import subprocess
    subprocess.check_call(['dot', '-Tpng', 'test.dot', '-o', '{}to{}.png'.format(cell_one, cell_two)])
    
    O = initialize_go(all_exp_itxn)
    second_enrichment = go_enrichment(O, second_cell)
    
    subprocess.check_call(['dot', '-Tpng', 'test.dot', '-o', '{}to{}.png'.format(cell_two, cell_one)])
    
    return(first_enrichment, second_enrichment)

def go_enrich_by_cell(all_protein, interaction_matrix, cell_id, lookup_table, cell_subset = [0,1,3,5,7]):
    
    # background gene set: all R/ all L
    all_exp_itxn, R_protein, L_protein = interaction_background(interaction_matrix)
    background_list = [symbol_to_entrezgene(R_protein, lookup_table),symbol_to_entrezgene(L_protein, lookup_table)]
    
    X_list = []
    P_list = []
    namespace_list = []
    for rl in [0,1]:
        X = pd.DataFrame()
        P = pd.DataFrame()
        namespace = pd.Series()
        for partner in cell_subset:
        
            O = initialize_go(background_list[rl])
            g = go_enrichment(O, symbol_to_entrezgene(all_protein[1, partner, rl], lookup_table))
            g.set_index('name', inplace = True)
            print(g.shape)
            X = pd.concat([X, g['x']], axis = 1, ignore_index = False, sort = False)
            P = pd.concat([P, g['p']], axis = 1, ignore_index = False, sort = False)
            namespace = namespace.append(g['namespace'])
        
        X.columns = cell_subset
        X.fillna(0, inplace = True)
        P.columns = cell_subset
        P.fillna(1, inplace = True)
        X_list.append(X)
        P_list.append(X)
        namespace_list.append(namespace)
    
    
    return(X_list, P_list, namespace_list)

def partition_by_namespace(X_list, P_list, namespace_list):
    xlist = []
    plist = []
    for rl in [0,1]:
        for namespace in namespace_list[rl].unique():
            term_list = namespace_list[rl].loc[namespace_list[rl] == namespace].index
            xlist.append(X_list[rl].loc[X_list[rl].index.isin(term_list)])
            plist.append(P_list[rl].loc[P_list[rl].index.isin(term_list)])
    return(xlist, plist)

def bar_go(goenrich_df, top_result = 20):
    '''
    bar plot layout of goenrich's dataframe result
    goenrich_df: goenrich output dataframe with p value, namespace, name, x
    top_result: how many of them do u want to show
    
    '''
    # group terms by namespace, which is cellular component, molecular_function, biological process
    g = goenrich_df.groupby('namespace')
    
    # create subplot
    f, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, figsize=(15,5))
    f.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)

    # initialize colormap based on range of p-value
    cmap = plt.cm.plasma
    norm = mpl.colors.Normalize(vmin=np.log10(goenrich_df['p'].min()), vmax=np.log10(goenrich_df['p'].max()))
    
    i = 0
    for name, group in g:
        # the original dataframe is already sorted. only show top results
        if group.shape[0] > top_result:
            group = group.sort_values(by = 'p', ascending = True).iloc[:top_result]
        else:
            group = group.sort_values(by = 'p', ascending = True)
        
        # make p-value become the color
        colors = cmap(norm(np.log10(group['p'].values)))
        
        y_pos = np.arange(group.shape[0])
        
        ax[i].barh(y_pos, group['x'], align='center', color = colors)
        
        # y-axis
        ax[i].set_yticks(y_pos)
        ax[i].set_yticklabels(group['name'])
        ax[i].invert_yaxis()  # labels read top-to-bottom
        
        # title
        ax[i].set_title(name)
        
        # xlabel
        ax[i].set_xlabel('count')
        i += 1
    
    # create color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # only needed for matplotlib < 3.1
    cbar = f.colorbar(sm)
    
    cbar.ax.set_ylabel('log10 p-value')
    