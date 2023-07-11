"""
    Run pairwise differential mean expression tests given dataset of batches, individuals, celltypes, and conditions (or species).
"""

import argparse
import scanpy as sc
import logging
import itertools
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from statsmodels.stats.multitest import fdrcorrection

# This version isnt on PyPI yet, so its jank
import sys
import os
sys.path.append('/home/ssm-user/Github/memento')
import memento.model.rna as rna

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)


def parse_arguments():

    parser = argparse.ArgumentParser(prog='Cell type DE')
    
    parser.add_argument('--h5ad_path', type=str)
    parser.add_argument('--ct_col', type=str)
    parser.add_argument('--donor_col', type=str)
    parser.add_argument('--condition_col', type=str)
    parser.add_argument('--batch_col', type=str)
    parser.add_argument('--filter_expr', type=float, default=0.02)
    parser.add_argument('--out_path', type=str, default=os.getcwd())
    
    return parser.parse_args()


def get_unique(adata, colname):
    
    return adata.obs[colname].unique()


if __name__ == '__main__':
    
    args = parse_arguments()
    
    adata = sc.read(args.h5ad_path)
    adata.obs['q'] = 0.07
    
    # Setup
    rna.MementoRNA.setup_anndata(
        adata=adata,
        q_column='q',
        label_columns=[args.condition_col, args.ct_col, args.batch_col, args.donor_col],
        num_bins=30)

    # Filter for some expression threshold
    expr_count = adata.X.mean(axis=0).A1
    adata = adata[:, expr_count > args.filter_expr]
    
    # Compute estimates for all groups of cells
    model = rna.MementoRNA(adata=adata)
    model.compute_estimate(estimand='mean', get_se=True)
    
    # Define the tests
    conditions = get_unique(adata, args.condition_col)
    cell_types = get_unique(adata, args.ct_col)
    
    # Loop over conditions
    for condition in conditions:
        
        # Loop over pairs of cell types
        for ct1, ct2 in itertools.combinations(cell_types, 2):
            
            # Create design matrix for this test
            # TODO: Make this a util function
            groups = [g for g in adata.uns['memento']['groups'] if condition in g and (ct1 in g or ct2 in g)]
            df = pd.DataFrame(index=groups)
            df['ct'] = df.index.str.split('^').str[2]
            df['batch'] = df.index.str.split('^').str[3]
            df['donor'] = df.index.str.split('^').str[4]
            cov_df = pd.get_dummies(df[['batch', 'donor']], drop_first=True).astype(float)
            ct_df = (df[['ct']]==ct1).astype(float)
            cov_df = sm.add_constant(cov_df)
            
            result = model.differential_mean(
                covariates=cov_df, 
                treatments=ct_df,
                family='quasiGLM',
                verbose=2,
                n_jobs=4)

            _, result['fdr'] = fdrcorrection(result['pval'])
            result.to_csv(Path(args.out_path, f'{condition}_{ct1}_{ct2}_memento.csv'))