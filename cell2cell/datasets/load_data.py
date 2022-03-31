# internal script to generate balf_balf_samples.pickle for tutorials
import os

import pandas as pd
import scanpy as sc

def load_balf(expression_data_path: str):
    """Load the BALF covid data set from Liao 2020 (https://doi.org/10.1038/s41591-020-0901-9)

    Parameters
    ----------
    expression_data_path : str
        full/path/to/balf_data_dir/. Should contain the sample counts as
        filtered_feature_bc_matrix.h5 as well as the metadata.txt file

    Returns
    -------
    balf_samples : Dict[str, AnnData]
        keys are the balf_samples, values are the raw UMI counts stored in an AnnData object
    """
    # load the data
    md = pd.read_csv(expression_data_path + 'metadata.txt', sep='\t')
    md.rename(columns = {'sample': 'Sample_ID', 'group': 'Context', 'celltype': 'cell_type', 'ID': 'cell_barcode'}, 
                    inplace = True)
    md['Context'] = md.Context.map({'HC': 'Healthy_Control', 'M': 'Moderate_Covid', 'S': 'Severe_Covid'}).tolist()
    md.set_index('cell_barcode', inplace = True)

    balf_samples = dict()
    for filename in os.listdir(expression_data_path):
        if filename.endswith('.h5'):
            sample = filename.split('_')[1]
            
            # subset and format metadata
            md_sample = md[md.Sample_ID == sample]
            md_sample.index = [cell_barcode.split('_')[0] + '-1' for cell_barcode in md_sample.index]
            
            adata = sc.read_10x_h5(expression_data_path + filename)
            adata = adata[md_sample.index, ]
            adata.obs = md_sample[['Sample_ID', 'Context', 'cell_type']]
            adata.var_names_make_unique()
            balf_samples[sample] = adata
    return balf_samples
