import scanpy as sc
import numpy as np
import logging

from memento.estimator.hypergeometric import (
    hg_mean, 
    hg_variance, 
    residual_variance, 
    fit_mv_regressor
)
from memento.estimator.sample import sample_mean
from memento.util import select_cells

class MementoRNA():
    """
        Class for performing differential exmpression testing on scRNA-seq data.
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        q_column: str
    ):
        self.adata = adata
        self.q_column = q_column
        self.estimands = ['mean', 'variance', 'correlation']
        
    
    @classmethod
    def setup_anndata(
        cls,
        adata: sc.AnnData,
        q_column:str,
        label_columns,
        **kwargs
    ):
        adata.uns['memento'] = {}
        adata.uns['memento']['q_column'] = q_column
        adata.uns['memento']['q'] = adata.obs[q_column].values
        
        logging.info(f'setup_anndata: creating groups')
        cls.create_groups(adata=adata,label_columns=label_columns, **kwargs)
        
        logging.info(f'setup_anndata: computing cell sizes')
        cls.compute_cell_size(adata=adata, **kwargs)

        
    @classmethod
    def create_groups(
        cls,
        adata: sc.AnnData,
        label_columns: list, 
        label_delimiter: str = '^'
    ):
        adata.obs['memento_group'] = 'sg' + label_delimiter
        for idx, col_name in enumerate(label_columns):
            adata.obs['memento_group'] += adata.obs[col_name].astype(str)
            if idx != len(label_columns)-1:
                adata.obs['memento_group'] += label_delimiter
    
        # Create a dict in the uns object
        adata.uns['memento']['label_columns'] = label_columns
        adata.uns['memento']['label_delimiter'] = label_delimiter
        adata.uns['memento']['groups'] = adata.obs['memento_group'].drop_duplicates().tolist()

        # Create slices of the data based on the group
        adata.uns['memento']['group_cells'] = {group:select_cells(adata, group) for group in adata.uns['memento']['groups']}

        # For each slice, get mean q
        adata.uns['memento']['group_q'] = {group:adata.uns['memento']['q'][(adata.obs['memento_group'] == group).values].mean() for group in adata.uns['memento']['groups']}
        

    @classmethod
    def compute_cell_size(
        cls,
        adata: sc.AnnData,
        use_raw: bool = True,
        filter_thresh:float = 0.07,
        trim_percent: float = 0.1,
        shrinkage: float = 0.5, 
        num_bins: int = 30):
        
        # Save the parameters
        adata.uns['memento']['size_factor_params'] = {
            'filter_thresh':filter_thresh,
            'trim_percent':trim_percent,
            'shrinkage':shrinkage,
            'num_bins':num_bins}
        
        # Compute naive size factors and UMI depth
        if use_raw and adata.raw:
            X = adata.raw.X
        else:
            X = adata.X
        naive_size_factor = X.sum(axis=1).A1
        umi_depth = np.median(naive_size_factor)
                
        # Compute residual variance with naive size factors
        m = sample_mean(X, size_factor=naive_size_factor)
        v = hg_variance(X, q=adata.uns['memento']['q'].mean(), size_factor=naive_size_factor)
        m[X.mean(axis=0).A1 < filter_thresh] = 0
        rv = residual_variance(m, v, fit_mv_regressor(m,v))
        
        # Select genes for normalization
        rv_ulim = np.quantile(rv[np.isfinite(rv)], trim_percent)
        rv[~np.isfinite(rv)] = np.inf
        rv_mask = (rv <= rv_ulim)
        mask = rv_mask
        adata.uns['memento']['least_variable_genes'] = adata.var.index[mask].tolist()
        
        # Re-estimate size factor
        size_factor = X.multiply(mask).sum(axis=1).A1
        if shrinkage > 0:
            size_factor += np.quantile(size_factor, shrinkage)
        size_factor = size_factor / np.median(size_factor)
        size_factor = size_factor * umi_depth
        adata.obs['memento_size_factor'] = size_factor