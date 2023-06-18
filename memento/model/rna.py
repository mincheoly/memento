import scanpy as sc
import memento.estimator.hypergeometric as hg
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
        
    
    def compute_cell_size(
        self,
        use_raw: bool = False,
        trim_percent: float = 0.1,
        shrinkage: float = 0.5, 
        num_bins: int = 30):
        
        
        
        
    
    
        