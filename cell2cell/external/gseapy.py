import os

from collections import defaultdict
from tqdm import tqdm

from types import ModuleType


def load_gmt(filename, backup_url=None, readable_name=False):
    '''Load a GMT file.

    Parameters
    ----------
    filename : str
        Path to the GMT file.

    backup_url : str, default=None
        URL to download the GMT file from if not present locally.

    readable_name : boolean, default=False
        If True, the pathway names are transformed to a more readable format.
        That is, removing underscores and pathway DB name at the beginning.

    Returns
    -------
    pathway_per_gene : dict
        Dictionary with genes as keys and pathways as values.
    '''
    from pathlib import Path

    path = Path(filename)
    if path.is_file():
        f = open(path, 'rb')
    else:
        if backup_url is not None:
            try:
                _download(backup_url, path)
            except ValueError:  # invalid URL
                print('Invalid filename or URL')
            f = open(path, 'rb')
        else:
            print('Invalid filename')

    pathway_per_gene = defaultdict(set)
    with f:
        for i, line in enumerate(f):
            l = line.decode("utf-8").split('\t')
            l[-1] = l[-1].replace('\n', '')
            l = [pw for pw in l if ('http' not in pw)]  # Remove website info
            pathway_name = l[0]
            if readable_name:
                pathway_name = ' '.join(pathway_name.split('_')[1:])
            for gene in l[1:]:
                pathway_per_gene[gene] = pathway_per_gene[gene].union(set([pathway_name]))
    return pathway_per_gene


def generate_lr_geneset(lr_list, complex_sep=None, lr_sep='^', pathway_per_gene=None, organism='human', pathwaydb='GOBP',
                        min_pathways=15, max_pathways=10000, readable_name=False, output_folder=None):
    '''Generate a gene set from a list of LR pairs.

    Parameters
    ----------
    lr_list : list
        List of LR pairs.

    complex_sep : str, default=None
        Separator of the members of a complex. If None, the ligand and receptor are assumed to be single genes.

    lr_sep : str, default='^'
        Separator of the ligand and receptor in the LR pair.

    pathway_per_gene : dict, default=None
        Dictionary with genes as keys and pathways as values.
        You can pass this if you are using different annotations than those
        available resources in `cell2cell.datasets.gsea_data.gsea_msig()`.

    organism : str, default='human'
        Organism for whom the DB will be loaded.
        Available options are {'human', 'mouse'}.

    pathwaydb: str, default='GOBP'
        Molecular Signature Database to load.
        Available options are {'GOBP', 'KEGG', 'Reactome'}

    min_pathways : int, default=15
        Minimum number of pathways that a LR pair can be annotated to.

    max_pathways : int, default=10000
        Maximum number of pathways that a LR pair can be annotated to.

    readable_name : boolean, default=False
        If True, the pathway names are transformed to a more readable format.

    output_folder : str, default=None
        Path to store the GMT file. If None, it stores the gmt file in the
        current directory.

    Returns
    -------
    lr_set : dict
        Dictionary with pathways as keys and LR pairs as values.
    '''
    # Check if the LR gene set is already available
    _check_pathwaydb(organism, pathwaydb)

    # Obtain annotations
    gmt_info = PATHWAY_DATA[organism][pathwaydb].copy()
    if output_folder is not None:
        gmt_info['filename'] = os.path.join(output_folder, gmt_info['filename'])
    if pathway_per_gene is None:
        pathway_per_gene = load_gmt(readable_name=readable_name, **gmt_info)

    # Dictionary to save the LR interaction (key) and the annotated pathways (values).
    pathway_sets = defaultdict(set)

    # Iterate through the interactions in the LR DB.
    for lr_label in lr_list:
        lr = lr_label.split(lr_sep)

        # Gene members of the ligand and the receptor in the LR pair
        if complex_sep is None:
            ligands = [lr[0]]
            receptors = [lr[1]]
        else:
            ligands = lr[0].split(complex_sep)
            receptors = lr[1].split(complex_sep)

        # Find pathways associated with all members of the ligand
        for i, ligand in enumerate(ligands):
            if i == 0:
                ligand_pathways = pathway_per_gene[ligand]
            else:
                ligand_pathways = ligand_pathways.intersection(pathway_per_gene[ligand])

        # Find pathways associated with all members of the receptor
        for i, receptor in enumerate(receptors):
            if i == 0:
                receptor_pathways = pathway_per_gene[receptor]
            else:
                receptor_pathways = receptor_pathways.intersection(pathway_per_gene[receptor])

        # Keep only pathways that are in both ligand and receptor.
        lr_pathways = ligand_pathways.intersection(receptor_pathways)
        for p in lr_pathways:
            pathway_sets[p] = pathway_sets[p].union([lr_label])

    lr_set = defaultdict(set)

    for k, v in pathway_sets.items():
        if min_pathways <= len(v) <= max_pathways:
            lr_set[k] = v
    return lr_set


def run_gsea(loadings, lr_set, output_folder, weight=1, min_size=15, permutations=999, processes=6,
             random_state=6, significance_threshold=0.05):
    '''Run GSEA using the LR gene set.

    Parameters
    ----------
    loadings : pandas.DataFrame
        Dataframe with the loadings of the LR pairs for each factor.

    lr_set : dict
        Dictionary with pathways as keys and LR pairs as values.
        LR pairs must match the indexes in the loadings dataframe.

    output_folder : str
        Path to the output folder.

    weight : int, default=1
        Weight to use for score underlying the GSEA (parameter p).

    min_size : int, default=15
        Minimum number of LR pairs that a pathway must contain.

    permutations : int, default=999
        Number of permutations to use for the GSEA. The total permutations
        will be this number plus 1 (this extra case is the unpermuted one).

    processes : int, default=6
        Number of processes to use for the GSEA.

    random_state : int, default=6
        Random seed to use for the GSEA.

    significance_threshold : float, default=0.05
        Significance threshold to use for the FDR correction.

    Returns
    -------
    pvals : pandas.DataFrame
        Dataframe containing the P-values for each pathway (rows)
        in each of the factors (columns).

    score : pandas.DataFrame
        Dataframe containing the Normalized Enrichment Scores (NES)
        for each pathway (rows) in each of the factors (columns).

    gsea_df : pandas.DataFrame
        Dataframe with the detailed GSEA results.
    '''
    import numpy as np
    import pandas as pd

    from statsmodels.stats.multitest import fdrcorrection
    from cell2cell.io.directories import create_directory

    create_directory(output_folder)
    gseapy = _check_if_gseapy()

    lr_set_ = lr_set.copy()
    df = loadings.reset_index()
    for factor in tqdm(df.columns[1:]):
        # Rank LR pairs of each factor by their respective loadings
        test = df[['index', factor]]
        test.columns = [0, 1]
        test = test.sort_values(by=1, ascending=False)
        test.reset_index(drop=True, inplace=True)

        # RUN GSEA
        gseapy.prerank(rnk=test,
                       gene_sets=lr_set_,
                       min_size=min_size,
                       weighted_score_type=weight,
                       processes=processes,
                       permutation_num=permutations,  # reduce number to speed up testing
                       outdir=output_folder + '/GSEA/' + factor, format='pdf', seed=random_state)

    # Adjust P-values
    pvals = []
    terms = []
    factors = []
    nes = []
    for factor in df.columns[1:]:
        p_report = pd.read_csv(output_folder + '/GSEA/' + factor + '/gseapy.gene_set.prerank.report.csv')
        pval = p_report['NOM p-val'].values.tolist()
        pvals.extend(pval)
        terms.extend(p_report.Term.values.tolist())
        factors.extend([factor] * len(pval))
        nes.extend(p_report['NES'].values.tolist())
    gsea_df = pd.DataFrame(np.asarray([factors, terms, nes, pvals]).T, columns=['Factor', 'Term', 'NES', 'P-value'])
    gsea_df = gsea_df.loc[gsea_df['P-value'] != 'nan']
    gsea_df['P-value'] = pd.to_numeric(gsea_df['P-value'])
    gsea_df['P-value'] = gsea_df['P-value'].replace(0., 1. / (permutations + 1))
    gsea_df['NES'] = pd.to_numeric(gsea_df['NES'])
    # Corrected P-value
    gsea_df['Adj. P-value'] = fdrcorrection(gsea_df['P-value'].values,
                                            alpha=significance_threshold)[1]
    gsea_df.to_excel(output_folder + '/GSEA/GSEA-Adj-Pvals.xlsx')

    pvals = gsea_df.pivot(index="Term", columns="Factor", values="Adj. P-value").fillna(1.)
    scores = gsea_df.pivot(index="Term", columns="Factor", values="NES").fillna(0)

    # Sort factors
    sorted_columns = [f for f in loadings.columns[1:] if (f in pvals.columns) & (f in scores.columns)]
    pvals = pvals[sorted_columns]
    scores = scores[sorted_columns]
    return pvals, scores, gsea_df


def _download(url, path):
    '''Function borrowed from scanpy

    https://github.com/scverse/scanpy/blob/2e98705347ea484c36caa9ba10de1987b09081bf/scanpy/readwrite.py#L936
    '''
    try:
        import ipywidgets
        from tqdm.auto import tqdm
    except ImportError:
        from tqdm import tqdm

    from urllib.request import urlopen, Request
    from urllib.error import URLError

    blocksize = 1024 * 8
    blocknum = 0

    try:
        req = Request(url, headers={"User-agent": "cell2cell-user"})

        try:
            open_url = urlopen(req)
        except URLError:
            print(
                'Failed to open the url with default certificates, trying with certifi.'
            )

            from certifi import where
            from ssl import create_default_context

            open_url = urlopen(req, context=create_default_context(cafile=where()))

        with open_url as resp:
            total = resp.info().get("content-length", None)
            with tqdm(
                unit="B",
                unit_scale=True,
                miniters=1,
                unit_divisor=1024,
                total=total if total is None else int(total),
            ) as t, path.open("wb") as f:
                block = resp.read(blocksize)
                while block:
                    f.write(block)
                    blocknum += 1
                    t.update(len(block))
                    block = resp.read(blocksize)

    except (KeyboardInterrupt, Exception):
        # Make sure file doesn’t exist half-downloaded
        if path.is_file():
            path.unlink()
        raise


def _check_if_gseapy() -> ModuleType:
    try:
        import gseapy

    except Exception:
        raise ImportError('gseapy is not installed. Please install it with: '
                          'pip install gseapy'
                          )
    return gseapy


def _check_pathwaydb(organism, pathwaydb):
    if organism not in PATHWAY_DATA.keys():
        raise ValueError('Invalid organism')

    if pathwaydb not in PATHWAY_DATA[organism].keys():
        raise ValueError('Invalid pathwaydb')


PATHWAY_DATA = {'human' : {'GOBP' : {'filename' : 'Human-2023-GOBP-msigdb-symbols.gmt',
                                     'backup_url' : 'https://zenodo.org/record/7593519/files/Human-2023-GOBP-msigdb-symbols.gmt'},
                           'KEGG' : {'filename' : 'Human-2023-KEGG-msigdb-symbols.gmt',
                                     'backup_url' : 'https://zenodo.org/record/7593519/files/Human-2023-KEGG-msigdb-symbols.gmt'},
                           'Reactome' : {'filename' : 'Human-2023-Reactome-msigdb-symbols.gmt',
                                         'backup_url' : 'https://zenodo.org/record/7593519/files/Human-2023-Reactome-msigdb-symbols.gmt'},
                           },
                'mouse' : {'GOBP' : {'filename' : 'Mouse-2023-GOBP-msigdb-symbols.gmt',
                                     'backup_url' : 'https://zenodo.org/record/7593519/files/Mouse-2023-GOBP-msigdb-symbols.gmt'},
                           'Reactome' : {'filename' : 'Mouse-2023-Reactome-msigdb-symbols.gmt',
                                         'backup_url' : 'https://zenodo.org/record/7593519/files/Mouse-2023-Reactome-msigdb-symbols.gmt'},
                           },
                }