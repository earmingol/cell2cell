from typing import Dict, Optional
import warnings 

import pandas as pd
import numpy as np
import itertools
from scipy.stats import gmean
from sklearn.decomposition import PCA
import meld

class CellComposition:
    """Accounting for the cell composition of the data.
    
    Each method provides a different approach to calculating a "composition score" indicative of the cell or cell type's relative composition/frequency. 
    Some scores are at single-cell resolution, others at the higher-order grouping (e.g., cell type or cluster). 
    """
    
    _method_map = {'gmean': gmean, 'mean': np.mean, 'median': np.median}
    
    def __init__(self, expr: pd.DataFrame, metadata: pd.DataFrame, manifold: Optional[pd.DataFrame] = None,
                random_state: Optional[int] = 888):
        """Get the cell composition of the data

        Parameters
        ----------
        expr : pd.DataFrame
            the expression matrix with cell barcodes as columns and gene IDs as the index
            should contain cells from all contexts in a single matrix (recommended batch-corrected)
        metadata : pd.DataFrame
            cell-level metadata, with index as cell barcodes 
            should include the samples/contexts (e.g., disease state or time point) and a higher-level grouping (e.g., cell type or cluster)
            if working with a single context, still include this metadata with a single identifier label
        manifold : pd.DataFrame, optional
            reduced dimension coordinates, recommended PCA, by default None
            index is the cell barcode (same as metadata), columns are each dimension
            if None, will calculate to 100 PCs if needed
        random_state : int, optional
            seed for reproducible outputs, by default 888
        """
        self.expr = expr
        self.metadata = metadata
        self.manifold = manifold
        self._check_barcodes()
        
        # calculated by score_composition 
        self.barcode_composition_score = None 
        self.aggregate_composition_score = None
        
    def _check_barcodes(self):
        """Ensures that barcodes are in same order in metadata, expression matrix, and manifold."""
        if not np.all(self.expr.columns == self.metadata.index):
            if len(set(self.expr.columns).difference(self.metadata.index)) == 0:
                self.expr = self.expr.loc[:, self.metadata.index]
            else:
                raise ValueError('Expression matrix and metadata matrix barcodes disagree')
        if self.manifold is not None:
            if not np.all(self.manifold.index == self.metadata.index):
                if len(set(self.manifold.index).difference(self.metadata.index)) == 0:
                    self.manifold = self.manifold.loc[self.metadata.index, :]
                else:
                    raise ValueError('Expression matrix and metadata matrix barcodes disagree')

    def run_pca(self, n_pcs: int = 100):
        """Gets the PC space of expression matrix

        Parameters
        ----------
        n_pcs : int, optional
            number of PCs for PCA, by default 100

        """
        pca_op = PCA(n_components = n_pcs, random_state=self.random_state)
        data_pca = pca_op.fit_transform(self.expr.T)

        # format
        self.manifold = pd.DataFrame(data_pca, 
                                    columns = ['PC_' + str(i) for i in range(1,n_pcs+1)], 
                                    index = self.expr.columns)
    
    def score_composition(self, score_type: str = '', **kwargs):
        """Get a "composition score" representative of the cell or cell type frequency"""
        if score_type == 'meld':
            self.score_composition_meld(**kwargs)
    
    def score_composition_meld(self, context_label = 'Sample_ID'):
        """Calculates the composition score using MELD (https://doi.org/10.1038/s41587-020-00803-5). 
        MELD provides single-cell resolution of relative likelihood of observing a given cell in one context vs all the other contexts. 
        Here, we run MELD with default parameters for each context vs rest, and retain the likelihood scores of the cell if it is actually present in that context. 

        Requires > 1 context. 

        Parameters
        ----------
        context_label : str, optional
            metadata column that separates cells by sample or context, by default 'Sample_ID'

        Returns
        -------
        composition_metadata : pd.DataFrame
            same as input metadata df, but with a column that contains a cell compositional score
        """
        if self.metadata[context_label].nunique() < 2:
            raise ValueError('MELD can only be used to score composition frequencies if there is > 1 context')
        if self.manifold is None:
            self.run_pca()

        meld_op = meld.MELD()
        sample_densities = meld_op.fit_transform(self.manifold, sample_labels=self.metadata[context_label])
        sample_likelihoods = meld.utils.normalize_densities(sample_densities)
        sample_likelihoods.index = self.metadata.index
        # TODO: add some normalization to the likelihoods? as more samples are added, the average likelihood decreases
        # can add this either here or at the end

        # retain scores from the actual sample the cell was measured in s.t. each single-cell has 1 likelihood score associated with it rather than n_context likelihood scores
        context_map = dict(zip(self.metadata.index, self.metadata[context_label]))
        likelihood_score = {}
        for barcode, context in context_map.items():
            likelihood_score[barcode] = sample_likelihoods.loc[barcode, context]
        self.barcode_composition_score = likelihood_score
    
#     def normalize_composition_score(self):
#         """Normalize the composition score between [0,1]"""
#         if min(self.barcode_composition_score.values()) < 0:
#             raise ValueError('Composition score should be > 0')
#         if max(self.barcode_composition_score.values()) > 1:#TODO: normalize b/w [0,1]
#             print('Internal: need to write an appropriate normalization function')
#             raise ValueError('incomplete code')

    def get_aggregate_composition_score(self, aggregation_method='median', fill_na: float = 0, 
                                            context_label = 'Sample_ID', cellgroup_label='cell_type'):
        """Aggregate the composition score by the context and higher-level cell grouping (e.g., cell type or cluster)

        Parameters
        ----------
        aggregation_method : str, optional
            method by which to aggregate the scores, by default 'mean'
            Options are:
            - 'median' : Median composition score across the cell types for a given context
            - 'mean' : Mean composition score across the cell types for a given context
            - 'gmean' : Geometric mean composition score across the cell types for a given context
        fill_na : float, optional
            value to fill NaN with after aggregation, by default 0
            will not fill if fill_na = None
            NaN arises due to a cell type not being scored as present in a context, which is why we default it to 0 composition 
        context_label : str, optional
            metadata column that separates cells by sample or context, by default 'Sample_ID'
        cellgroup_label : str, optional
            metadata column that separates cells by higher-level cell grouping (e.g., cell type or cluster), by default 'cell_type'

        Raises
        ------
        ValueError
            _description_
        """
        if self.aggregate_composition_score is not None:
            warnings.warn('An aggregate composition score was already calculated, overwriting')
        if self.barcode_composition_score is None:
            raise ValueError('Must calculate a single-cell resolution composition score to aggregate on')

        composition_metadata = self.metadata.copy()
        composition_metadata['Composition_Score'] = pd.Series(composition_metadata.index).map(self.barcode_composition_score).tolist()
        
        composition_metadata = composition_metadata.groupby([context_label, cellgroup_label]).Composition_Score.apply(self._method_map[aggregation_method])
        composition_metadata = pd.DataFrame(composition_metadata).reset_index()
        
        if fill_na is not None:
            composition_metadata.Composition_Score.fillna(value = fill_na, inplace = True)

        self.aggregate_composition_score = {}
        for context in composition_metadata[context_label].unique():
            self.aggregate_composition_score[context] = \
                        composition_metadata[composition_metadata[context_label] == context]\
                        [[cellgroup_label, 'Composition_Score']].reset_index(drop = True).set_index(cellgroup_label)
    @staticmethod
    def get_pairwise_composition_score(composition_score: pd.DataFrame, method: str ='gmean'):
        """Calculate a pairwise composition score between two cell types from their individual composition scores.

        Parameters
        ----------
        composition_score : pd.DataFrame
            data frame of aggregated composition scores, with index as the cell type/cluster and one column containing the composition score for that cell type
            (for a single context)
        method : str, optional
            method to get pairwise composition score, by default 'gmean'
            Options are:
            - 'gmean' : Geometric mean composition score between the cell types
            - 'median' : Median composition score between the cell types
            - 'mean' : Mean composition score between the cell types

        Returns
        -------
        pairwise_score: pd.DataFrame
            the pairwise composition scores
        """
        
        pairwise_score = pd.DataFrame(CellComposition._method_map[method](np.meshgrid(composition_score, composition_score), axis = 0).flatten(), 
                            columns = ['Composition_Score'], 
                             index = ['-'.join(x) for x in itertools.product(composition_score.index, repeat = 2)])
        return pairwise_score

def normalize_lr_scores_by_max(lr_communication_scores: Dict[str, pd.DataFrame]):
    """Normalize the LR communication scores to the maximum one across all contexts. 
    This is done to create equal scaling of scores b/w LR communication and cell composition scores
    Only need if going to use the "scale_by_composition" function. 

    Parameters
    ----------
    lr_communication_scores : Dict[str, pd.DataFrame]
        keys are contexts
        values are a communication matrix with columns as (Sender-Receiver) pairs and rows as (Ligand&Receptor) pairs and values as the
        ligand-receptor communication score calculated from gene expression

    Returns
    -------
    lr_communication_scores
        same as input, but with normalized values
    """
    max_score = 0
    for lcs in lr_communication_scores.values():
        if lcs.min().min() < 0:
            raise ValueError('This function is specifically designed for non-negative LR communication scores')
        cur_score = lcs.max().max()
        if cur_score > max_score:
            max_score = cur_score
    
    if max_score <= 1: 
        warnings.warn('LR scores already lie between [0,1]')

    lr_communication_scores = {context: lcs/max_score for context, lcs in lr_communication_scores.items()}

    return lr_communication_scores

def scale_by_composition(lr_communication_score: pd.DataFrame, 
                         pairwise_composition_score: pd.DataFrame, 
                        composition_weight: float = 0.25):
    """Scales a LR communication score by the respective pairwise communication scores

    Parameters
    ----------
    lr_communication_score : pd.DataFrame
        a communication matrix with columns as (Sender-Receiver) pairs and rows as (Ligand&Receptor) pairs and values as the
        ligand-receptor communication score calculated from gene expression
    pairwise_composition_score : Dict[str, pd.DataFrame]
        the composition scores between pairs of cells
        index is the (Sender-Receiver) pair that should match the columns of the lr_communication_score
        see CellComposition.get_pairwise_composition_score for details
    composition_weight : float, optional
        weighting of composition score relative to communication score

    Returns
    -------
    communication_score
        the communication matrix with LR communication scores weighted by pairwise cell composition scores
    """
    
    # checks
    if (composition_weight > 1) or (composition_weight < 0):
        raise ValueError('The composition weight must lie between [0,1]')
    
    for df in [lr_communication_score, pairwise_composition_score]:
        if (df.min().min() < 0) or (df.max().max() > 1):
            raise ValueError('The communication and composition scores are expected to lie between [0,1]')
    
    if len(set(pairwise_composition_score.index).difference(lr_communication_score.columns)):
        raise ValueError('The sender-receiver cell pairs do not match between LR communication and pairwise composition scores')
    
    # ensure the order is matched
    pairwise_composition_score = pairwise_composition_score.loc[lr_communication_score.columns,:] 
    

    # do the calculation
    communication_score = pd.DataFrame(composition_weight*pairwise_composition_score.values + \
                                   ((1-composition_weight)*lr_communication_score).T.values, 
                                  index = lr_communication_score.columns, 
                                  columns = lr_communication_score.index).T/2
    
    return communication_score
    