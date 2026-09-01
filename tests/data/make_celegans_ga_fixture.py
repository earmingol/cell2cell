# -*- coding: utf-8 -*-

'''Regenerates the *C. elegans* genetic-algorithm fixture from the reference repository.

The fixture pins `cell2cell.analysis.genetic_algorithm` against the published analysis of
[Armingol et al. (2022)](https://doi.org/10.1371/journal.pcbi.1010715), whose 100 genetic-algorithm
executions are stored in
[LewisLabUCSD/Celegans-cell2cell](https://github.com/LewisLabUCSD/Celegans-cell2cell) under
`data/GA-Bray-Curtis/`. It holds the selection each execution converged to, the ligand-receptor
pairs those selections are indexed against, and the list of pairs the paper reports, which is
everything needed to check the consensus without downloading the expression data.

Run it only to refresh the fixture:

    python tests/data/make_celegans_ga_fixture.py

Requires a network connection, and writes `celegans_ga_consensus.npz` next to this file.
'''

import json
import pathlib
import urllib.request

import numpy as np
import pandas as pd

import cell2cell as c2c

BASE = 'https://raw.githubusercontent.com/LewisLabUCSD/Celegans-cell2cell/master/data/'
LISTING = ('https://api.github.com/repos/LewisLabUCSD/Celegans-cell2cell/contents/'
           'data/GA-Bray-Curtis')
INTERACTION_COLUMNS = ('Ligand_symbol', 'Receptor_symbol')
OUTPUT = pathlib.Path(__file__).parent / 'celegans_ga_consensus.npz'


def read_json(url):
    return json.loads(urllib.request.urlopen(url).read())


def final_masks():
    '''The selection each stored execution converged to, i.e. its last run.'''
    names = sorted(entry['name'] for entry in read_json(LISTING))
    masks = []
    for name in names:
        execution = read_json(BASE + 'GA-Bray-Curtis/' + name)
        last = max((k for k in execution if k.startswith('run')), key=lambda k: int(k[3:]))
        masks.append(np.asarray(execution[last]['ppi_data'], dtype=np.uint8))
    return np.asarray(masks), names


def pair_pool():
    '''The pairs the masks are indexed against.

    Positional masks are only meaningful against the ordering the original analysis used,
    which is the order of the curated table itself, restricted to the pairs whose ligand
    and receptor are both measured.
    '''
    rnaseq = c2c.io.load_rnaseq(rnaseq_file=BASE + 'RNA-Seq/Celegans_RNASeqData_Cell.xlsx',
                                gene_column='symbol', drop_nangenes=True,
                                log_transformation=False, format='auto', verbose=False)
    curated = c2c.io.load_table(BASE + 'PPI-Networks/Celegans-Curated-LR-pairs.xlsx',
                                format='auto', verbose=False)
    curated = c2c.preprocessing.remove_ppi_bidirectionality(
        curated, INTERACTION_COLUMNS, verbose=False)
    return c2c.preprocessing.preprocess_ppi_data(
        ppi_data=curated, interaction_columns=INTERACTION_COLUMNS,
        rnaseq_genes=list(rnaseq.index), upper_letter_comparison=False, verbose=False)


def main():
    masks, names = final_masks()
    pool = pair_pool()
    if masks.shape[1] != len(pool):
        raise RuntimeError('The stored masks are {} long but the pool holds {} pairs'
                           .format(masks.shape[1], len(pool)))

    selected = pd.read_csv(BASE + 'PPI-Networks/Celegans-GA-BrayCurtis-Selected-LR-pairs.csv')

    np.savez_compressed(
        OUTPUT,
        masks=masks,
        ligands=pool['A'].values.astype(str),
        receptors=pool['B'].values.astype(str),
        paper_ligands=selected['Ligand_symbol'].values.astype(str),
        paper_receptors=selected['Receptor_symbol'].values.astype(str),
        executions=np.asarray(names, dtype=str),
    )
    print('{}: {} executions x {} pairs, {} pairs in the published list, {:.1f} kB'
          .format(OUTPUT.name, masks.shape[0], masks.shape[1], len(selected),
                  OUTPUT.stat().st_size / 1e3))


if __name__ == '__main__':
    main()
