from cell2cell.external.gseapy import _check_pathwaydb, load_gmt, PATHWAY_DATA


def gsea_msig(organism='human', pathwaydb='GOBP', readable_name=False):
    '''Load a MSigDB from a gmt file

    Parameters
    ----------
    organism : str, default='human'
        Organism for whom the DB will be loaded.
        Available options are {'human', 'mouse'}.

    pathwaydb: str, default='GOBP'
        Molecular Signature Database to load.
        Available options are {'GOBP', 'KEGG', 'Reactome'}

    readable_name : boolean, default=False
        If True, the pathway names are transformed to a more readable format.
        That is, removing underscores and pathway DB name at the beginning.

    Returns
    -------
    pathway_per_gene : defaultdict
        Dictionary containing all genes in the DB as keys, and
        their values are lists with their pathway annotations.
    '''
    _check_pathwaydb(organism, pathwaydb)

    pathway_per_gene = load_gmt(readable_name=readable_name, **PATHWAY_DATA[organism][pathwaydb])
    return pathway_per_gene