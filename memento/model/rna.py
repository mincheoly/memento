import scanpy as sc
import memento.estimator.hypergeometric as hg
import memento.estimator.sample as sp
import memento.util as util

class MementoRNA():
    """
        Class for performing differential expression testing on scRNA-seq data.
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
    def create_groups(
        adata: sc.AnnData,
        label_columns: list, 
        label_delimiter: str = '^'
    ):
        adata.obs['memento_group'] = 'sg' + label_delimiter
        for idx, col_name in enumerate(label_columns):
            adata.obs['memento_group'] += obs[col_name].astype(str)
            if idx != len(label_columns)-1:
                adata.obs['memento_group'] += label_delimiter
    
        # Create a dict in the uns object
        adata.uns['memento']['label_columns'] = label_columns
        adata.uns['memento']['label_delimiter'] = label_delimiter
        adata.uns['memento']['groups'] = adata.obs['memento_group'].drop_duplicates().tolist()
        adata.uns['memento']['q'] = adata.obs[adata.uns['memento']['q_column']].values

        # Create slices of the data based on the group
        adata.uns['memento']['group_cells'] = {group:util._select_cells(adata, group) for group in adata.uns['memento']['groups']}

        # For each slice, get mean q
        adata.uns['memento']['group_q'] = {group:adata.uns['memento']['q'][(adata.obs['memento_group'] == group).values].mean() for group in adata.uns['memento']['groups']}
        

    @classmethod
    def compute_cell_size(
        adata: sc.AnnData,
        use_raw: bool = True,
        filter_thresh:float = 0.07,
        trim_percent: float = 0.1,
        shrinkage: float = 0.5, 
        num_bins: int = 30):
        
        # Compute naive size factors and UMI depth
        X = self.adata.raw.X if use_raw else adata.X
        naive_size_factor = X.sum(axis=1).A1
        umi_depth = np.median(naive_size_factor)
                
        # Compute residual variance with naive size factors
        m = sp.mean(X, size_factor=naive_size_factor)
        v = hg.variance(X, q=adata.uns['memento']['q'].mean(), size_factor=naive_size_factor)
        m[X.mean(axis=0).A1 < filter_mean_thresh] = 0
        rv = hg.residual_variance(m, v, hg.fit_mv_regressor(m,v))
        
        # Select genes for normalization
        rv_ulim = np.quantile(rv[np.isfinite(rv)], trim_percent)
        rv[~np.isfinite(rv)] = np.inf
        rv_mask = (all_res_var <= rv_ulim)
        mask = rv_mask
        adata.uns['memento']['least_variable_genes'] = adata.var.index[mask].tolist()
        
        # Re-estimate size factor
        size_factor = X.multiply(mask).sum(axis=1).A1
        if shrinkage > 0:
            size_factor += np.quantile(size_factor, shrinkage)
        size_factor = size_factor / np.median(size_factor)
        size_factor = size_factor * umi_depth
        adata.obs['memento_size_factor'] = size_factor