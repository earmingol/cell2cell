
import numpy as np
import pandas as pd


def get_cutoffs(RNAseq_data, type = 'percentile', val = 0.8, verbose = True, cutoff_data = None):
    if verbose:
        print("Calculating cutoffs for gene abundances")
    cutoffs = pd.DataFrame()
    if type == 'percentile':
        cutoffs['cutoff'] = RNAseq_data.quantile(val, axis=1)
        cutoffs['gene'] = RNAseq_data['gene_id']
    elif type == 'matlab':
        if cutoff_data is None:
            raise("Cutoff data has to be provided to use matlab cutoffs")
        gene_header = RNAseq_data.columns[0]
        cutoffs = cutoff_data[[gene_header, 'K=38']]
        cutoffs.columns = [gene_header, 'cutoff']
    else:
        raise ValueError(type + ' is not a valid cutoff')
    return cutoffs