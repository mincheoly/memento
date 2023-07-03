import scanpy as sc
import numpy as np
import pandas as pd
import logging
import scipy.stats as stats

from collections import OrderedDict
from joblib import Parallel, delayed
from functools import partial

from memento.estimator.hypergeometric import (
    hg_mean, 
    hg_sem_for_gene,
    hg_variance, 
    hg_sev_for_gene,
    residual_variance, 
    fit_mv_regressor
)
from memento.estimator.sample import sample_mean, sample_variance
from memento.util import select_cells, fit_nb, lrt_nb, meta_wls

class MementoRNA():
    """
        Class for performing differential expression testing on scRNA-seq data.
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        layer=None,
    ):
        self.adata = adata
        self.layer = layer
        self.mean_estimator_names = [
            'mean',
            'log_mean',
            'log1p_mean',
        ]
        self.var_estimator_names = [
            'var',
            'log_var',
            'resvar',
        ]
        self.corr_estimator_names = [
            'corr',
            'se_corr',
        ]
        self.estimator_group_names = [
            'total_umi',
            'cell_count'
        ]
        self.estimand_mapping = {
            'mean':self.mean_estimator_names,
            'var':self.var_estimator_names,
            'corr':self.corr_estimator_names
        }
        self.mv_regressors = {}
        self.estimates = {}
        
    
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
        adata.uns['memento']['q'] = adata.obs[q_column]
        
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
        adata.obs['memento_group'] = 'memento_group' + label_delimiter
        for idx, col_name in enumerate(label_columns):
            adata.obs['memento_group'] += adata.obs[col_name].astype(str)
            if idx != len(label_columns)-1:
                adata.obs['memento_group'] += label_delimiter
    
        # Create a dict in the uns object
        adata.uns['memento']['label_columns'] = label_columns
        adata.uns['memento']['label_delimiter'] = label_delimiter
        adata.uns['memento']['groups'] = adata.obs['memento_group'].drop_duplicates().tolist()

        # Create slices of the data based on the group
        adata.uns['memento']['group_barcodes'] = {group:select_cells(adata, group) for group in adata.uns['memento']['groups']}

        # For each slice, get mean q
        adata.uns['memento']['group_q'] = {group:adata.uns['memento']['q'][adata.uns['memento']['group_barcodes'][group]].values.mean() for group in adata.uns['memento']['groups']}
        

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
        # size_factor = size_factor * umi_depth
        adata.obs['memento_size_factor'] = size_factor
    
    
    def subset_matrix(self, barcodes, genes):
        """ Get a subset of the expr matrix for specific barcodes and genes. """
        subset = self.adata[barcodes, genes]
        if self.layer is not None:
            return subset.layers[self.layer]
        else:
            return subset.X
        
    
    def compute_estimate(
        self, 
        estimand,
        get_se=False,
        gene_list=None, 
        gene_pairs=None,
        n_boot=5000,
        n_jobs=1,
        verbose=0):
        """
            Compute the estimand.
        """ 
        
        # Make sure estimand is one of mean, var, corr
        if estimand not in ['mean', 'var', 'corr']:
            raise ValueError(f'estimand must be one of [mean, var, corr], given {estimand}')
                
        # Setup a dictionary to hold the estimates
        estimator = self.estimand_mapping[estimand] + self.estimator_group_names
        if get_se:
            estimator += ['se_' + est for est in estimator]
        estimates = {est:OrderedDict() for est in estimator}
        
        logging.info(f'compute_estimate: running estimators for {estimator}')
        
        if gene_list is None:
            logging.info(f'compute_estimate: gene_list is None, using all genes in AnnData object...')
            gene_list = self.adata.var.index.tolist()
        
        for group, barcodes in self.adata.uns['memento']['group_barcodes'].items():
            
            data = self.subset_matrix(barcodes, gene_list)
            sf = self.adata.obs.loc[barcodes]['memento_size_factor'].values
            q = self.adata.uns['memento']['group_q'][group]
            
            # Compute global quantities
            estimates['total_umi'][group] = data.sum()
            estimates['cell_count'][group] = data.shape[0]
            
            # Compute estimand specific quantities
            if estimand == 'mean':

                estimates['mean'][group] = hg_mean(X=data, size_factor=sf)
                estimates['log_mean'][group] = np.log(estimates['mean'][group])
                estimates['log1p_mean'][group] = np.log(estimates['mean'][group]+1)

                if get_se:
                    
                    logging.info(f'compute_estimate: getting bootstrap mean SE for using {n_jobs} parallel jobs')
                    gene_tasks = []
                    for idx, gene in enumerate(gene_list):

                        gene_tasks.append(partial(
                            hg_sem_for_gene,
                            X=data[:, idx],
                            q=q,
                            approx_size_factor=sf,
                            num_boot=n_boot,
                            group_name=group,
                        ))

                    results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)() for func in gene_tasks)
                    sem, selm, sel1pm = zip(*results)
                    estimates['se_mean'][group] = sem
                    estimates['se_log_mean'][group] = selm
                    estimates['se_log1p_mean'][group] = sel1pm

            elif estimand == 'var':

                v = hg_variance(X=data, q=q, size_factor=sf, group_name=group)
                v[v<=0] = np.nan

                # Fit mean-variance relationship
                m = estimates['mean'][group]
                self.mv_regressors[group] = fit_mv_regressor(m,v)
                rv = residual_variance(m, v, regressor)

                estimates['var'][group] = v
                estimates['log_var'] = np.log(v)
                estimates['resvar'][group] = rv
                estimates['log_resvar'][group] = np.log(rv)

                if get_se:

                    gene_tasks = []
                    for idx, gene in enumerate(gene_list):

                        gene_tasks.append(partial(
                            hg_sev_for_gene,
                            X=data[:, idx],
                            q=q,
                            approx_size_factor=sf,
                            mv_fit=self.mv_regressors[group],
                            num_boot=n_boot,
                            group_name=group,
                        ))

                    results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)() for func in gene_tasks)
                    sev, selv, serv, selrv = zip(*results)
                    estimates['se_var'][group] = sev
                    estimates['se_log_var'][group] = selv
                    estimates['se_resvar'][group] = serv
                    estimates['se_log_resvar'][group] = selrv

            else:

                raise NotImplementedError(f'compute_estimate: {est} estimator not yet implemented!')
                    
        for est,res in estimates.items():
            
            base_name = est.split('se_')[-1]
            if base_name in self.mean_estimator_names + self.var_estimator_names:
                
                self.estimates[est] = pd.DataFrame(res, index=gene_list).T
                
            elif est in self.estimator_group_names:
                
                self.estimates[est] = pd.DataFrame(res, index=[est]).T
                
            else:
                
                logging.warning(f'compute_estimates: storage for {base_name} not yet implemented!')
    

    def differential_mean(
        self, 
        covariates: pd.DataFrame,
        treatments: pd.DataFrame,
        treatment_for_gene: dict = None,
        family: str = 'NB',
        n_jobs=1,
        verbose=0):
        """ Perform differential expression using the given design matrix. """

        # Index the estimates by the groups actually present
        groups_in_test = covariates.index.tolist()
        test_estimates = {est:res.loc[groups_in_test] for est,res in self.estimates.items()}

        if family == 'NB':
            # Transform mean estimate to "pseudobulk"
            expr = test_estimates['mean']*test_estimates['total_umi'].values
            expr_se = test_estimates['sem']*test_estimates['cell_count'].values**2

            # Transform standard error to weights
            weights = np.sqrt(1/expr_se).replace([-np.inf, np.inf], np.nan)
            mean_weight = np.nanmean(weights)
            weights /= mean_weight
            weights = weights.fillna(1.0)            

            # Construct the tests
            tests = []  
            for idx, gene in enumerate(expr.columns):
                
                if treatment_for_gene is not None:
                    if gene in treatment_for_gene: # Get treatments for this gene
                        treatment_list = treatment_for_gene[gene]
                    else: # Pass this gene
                        continue
                else: # Default, get all pairwise treatment-gene tests
                    treatment_list = treatments.columns
                    
                for t in treatment_list:
                    
                    design_matrix = pd.concat([covariates, treatments[[t]]], axis=1)
                    
                    tests.append(
                        partial(
                            lrt_nb,
                            endog=expr.iloc[:, [idx]], 
                            exog=design_matrix,
                            exog0=covariates, 
                            offset=np.log(test_estimates['total_umi']['total_umi'].values), 
                            weights=weights.iloc[:, idx], 
                            gene=gene, 
                            t=t))
                    
            # Compute the tests in parallel
            result = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)() for func in tests)
            return pd.DataFrame(result, columns=['gene', 'treatment', 'coef', 'pval']).set_index('gene')
    
        if family == 'WLS':

            test_estimates['log1p_mean'] = np.log(test_estimates['mean']+1)
            l = np.log((test_estimates['mean']+1) - test_estimates['sem'])
            u = np.log((test_estimates['mean']+1) + test_estimates['sem'])
            test_estimates['log1p_sem'] = (u-l)/2

            expr = test_estimates['log1p_mean']
            expr_sem = test_estimates['log1p_sem']
            min_sem = expr_sem[expr_sem > 0].min(axis=0)
            for col in expr_sem.columns:
                expr_sem[col] = expr_sem[col].replace(0.0, min_sem[col])
            
            # Construct the tests
            tests = []  
            for idx, gene in enumerate(expr.columns):
                
                if treatment_for_gene is not None:
                    if gene in treatment_for_gene: # Get treatments for this gene
                        treatment_list = treatment_for_gene[gene]
                    else: # Pass this gene
                        continue
                else: # Default, get all pairwise treatment-gene tests
                    treatment_list = treatments.columns
                    
                for t in treatment_list:
                    
                    design_matrix = pd.concat([covariates, treatments[[t]]], axis=1)
                    
                    tests.append(
                        partial(
                            meta_wls,
                            y=expr.iloc[:, [idx]].values, 
                            X=design_matrix.values,
                            v=expr_sem.iloc[:, [idx]].values**2,
                            gene=gene, 
                            t=t))
                    
            # Compute the tests in parallel
            result = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)() for func in tests)
            return pd.DataFrame(result, columns=['gene', 'treatment', 'coef', 'pval']).set_index('gene')
            

            
            
            
            
            
            
            
            
                    
                    
                
                
        