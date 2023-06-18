import scipy.stats as stats
import numpy as np
import time
import itertools
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection


def bin_size_factor(size_factor, num_bins=30):
    """ Bin the size factors to speed up bootstrap. """
    
    binned_stat = stats.binned_statistic(size_factor, size_factor, bins=num_bins, statistic='mean')
    bin_idx = np.clip(binned_stat[2], a_min=1, a_max=binned_stat[0].shape[0])
    approx_sf = binned_stat[0][bin_idx-1]
    max_sf = size_factor.max()
    approx_sf[size_factor == max_sf] = max_sf
    
    return approx_sf



def _select_cells(adata, group):
    """ Slice the data horizontally. """

    cell_selector = (adata.obs['memento_group'] == group).values

    return adata.X[cell_selector, :].tocsc()


def _get_gene_idx(adata, gene_list):
    """ Returns the indices of each gene in the list. """

    return np.array([np.where(adata.var.index == gene)[0][0] for gene in gene_list]) # maybe use np.isin


def _fdrcorrect(pvals):
    """
    Perform FDR correction with nan's.
    """

    fdr = np.ones(pvals.shape[0])
    _, fdr[~np.isnan(pvals)] = fdrcorrection(pvals[~np.isnan(pvals)])
    return fdr