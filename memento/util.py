import scipy.stats as stats
import numpy as np
import time
import itertools
import scipy as sp
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
from pymare import estimators



def bin_size_factor(size_factor, num_bins=30):
    """ Bin the size factors to speed up bootstrap. """
    
    binned_stat = stats.binned_statistic(size_factor, size_factor, bins=num_bins, statistic='mean')
    bin_idx = np.clip(binned_stat[2], a_min=1, a_max=binned_stat[0].shape[0])
    approx_sf = binned_stat[0][bin_idx-1]
    max_sf = size_factor.max()
    approx_sf[size_factor == max_sf] = max_sf
    
    return approx_sf


def select_cells(adata, group):
    """ Slice the data horizontally. """

    # cell_selector = (adata.obs['memento_group'] == group).values

    # return adata.X[cell_selector, :].tocsc()
    
    cell_selector = adata.obs['memento_group'][adata.obs['memento_group'] == group].index.tolist()
    
    return cell_selector


def get_gene_idx(adata, gene_list):
    """ Returns the indices of each gene in the list. """

    return np.array([np.where(adata.var.index == gene)[0][0] for gene in gene_list]) # maybe use np.isin


def fdrcorrect(pvals):
    """
        Perform FDR correction with nan's.
    """

    fdr = np.ones(pvals.shape[0])
    _, fdr[~np.isnan(pvals)] = fdrcorrection(pvals[~np.isnan(pvals)])
    return fdr


def fit_nb(endog, exog, offset, alpha=None, weights=None):
    """
        Fit a negative binomial GLM using Poisson as a starting guess
    """
    

    if alpha is None:
        # Fit a preliminary Poisson model
        poi = sm.GLM(
            endog,
            exog, 
            offset=offset,
            family=sm.families.Poisson()).fit()

        # Estimate alpha
        mu = poi.predict()
        resid = poi.resid_response
        df_resid=poi.df_resid

        alpha = ((resid**2 / mu - 1) / mu).sum() / df_resid
        alpha = max(alpha, 1e-5)
        alpha = min(alpha, 10)

    # Fit a GLM
    nb = sm.GLM(
        endog,
        exog, 
        offset=offset,
        var_weights=weights,
        family=sm.families.NegativeBinomial(alpha=alpha))\
        .fit()
    
    return alpha, nb


def lrt_nb(endog, exog, exog0, offset, weights=None, gene=None, t=None):
    """
        Perform a likelihood ratio test using NB GLM.
    """
    
    nb_model = sm.NegativeBinomial(
        endog=endog,
        exog=exog, 
        offset=offset,
        weights=weights).fit()
    
    return (gene, t, nb_model.params[-1], nb_model.pvalues[-1])

#     try:
#         alpha, fit = fit_nb(
#             endog=endog,
#             exog=exog, 
#             offset=offset,
#             weights=weights)
#         _, res_fit = fit_nb(
#             endog=endog,
#             exog=exog0, 
#             offset=offset,
#             weights=weights,
#             alpha=alpha)
#     except:
#         return((gene, t, 0, 1))

#     pv = stats.chi2.sf(-2*(res_fit.llf - fit.llf), df=res_fit.df_resid-fit.df_resid)
#     return((gene, t, fit.params[-1], pv))


def meta_wls(y, X, v, gene=None, t=None):
    try:
        dsl = estimators.WeightedLeastSquares()
        dsl.fit(y=y, X=X, v=v)
        coef = float(dsl.summary().get_fe_stats()['est'][-1])
        se = float(dsl.summary().get_fe_stats()['se'][-1])
        p = float(dsl.summary().get_fe_stats()['p'][-1])
        
        # model = sm.OLS(y, X).fit()
        # coef = model.params[-1]
        # p = model.pvalues[-1]
    except:
        return((gene, t, 0, 0, 1))
    
    return ((gene, t, coef, se, p))