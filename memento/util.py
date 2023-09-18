import scipy.stats as stats
import numpy as np
import time
import itertools
import scipy as sp
import statsmodels.api as sm
import logging
from statsmodels.stats.multitest import fdrcorrection
from scipy.optimize import minimize_scalar, minimize
from sklearn.linear_model import LinearRegression


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


def lrt_nb(endog, exog, exog0, offset, weights=None, dispersion=None, gene=None, t=None):
    """
        Perform a likelihood ratio test using NB GLM.
    """
    
    if dispersion is None:
        try:
            alpha, fit = fit_nb(
                endog=endog,
                exog=exog, 
                offset=offset,
                weights=weights)
            _, res_fit = fit_nb(
                endog=endog,
                exog=exog0, 
                offset=offset,
                weights=weights,
                alpha=alpha)
        except:
            return((gene, t, 0, 1))
    else:
        try:
            _, fit = fit_nb(
                endog=endog,
                exog=exog, 
                offset=offset,
                weights=weights,
                alpha=dispersion)
            _, res_fit = fit_nb(
                endog=endog,
                exog=exog0, 
                offset=offset,
                weights=weights,
                alpha=dispersion)
        except:
            return((gene, t, 0, 1))
            
    # pv = stats.chi2.sf(-2*(res_fit.llf - fit.llf), df=res_fit.df_resid-fit.df_resid)
    pv = 0.5#fit.pvalues[t]
    return((gene, t, fit.params[-1], pv))


def meta_wls(y, X, v, n,gene=None, t=None):
    
    # try:
    if X.shape[0] < 2:
        return ((gene, t, np.nan, np.nan, 1))
        
    lm = LinearRegression(fit_intercept=False)
    lm.fit(X,y.ravel(), n.ravel())
    coef = float(lm.coef_[-1])
    W = 1/(v.ravel())
    se = np.sqrt(np.diag(np.linalg.pinv(X.T@np.diag(W)@X))[-1])
    p = 2*stats.norm.sf(np.abs(coef/se))
        
    # except:
    #     return((gene, t, 0, 0, 1))
    
    return ((gene, t, coef, se, p))


def nb_var_func(x, alpha):

    return (x+alpha*x**2)


def dispersion_objective(alpha, m, v, multiplier):

    return ((np.log(v*multiplier) - np.log(nb_var_func(m, alpha)))**2).mean()


def get_nb_sample_dispersions(expr, expr_var):
    
    num_groups = expr.shape[0]
    sample_dispersions = np.zeros(num_groups)

    for i in range(num_groups):

        m = expr[i]
        v = expr_var[i]

        lowexpr = m[m < np.quantile(m, 0.3)]
        lowexpr_var = v[m < np.quantile(m, 0.3)]
        multiplier = np.median((lowexpr/lowexpr_var)) # intercept on log-log plot to match Poisson assumption

        sample_dispersions[i] = minimize_scalar(
            lambda x: dispersion_objective(x, m=m, v=v, multiplier=multiplier), 
            bounds=[0, 10]).x
    
    return sample_dispersions


def wald_nb(
    endog, 
    exog, 
    offset,
    weights=None, 
    sample_dispersion=None,
    gene_dispersion=None,
    gene=None, 
    t=None):
    """
        Perform a Wald ratio test using NB GLM.
    """
    
    try:
        fit = sm.GLM(
            endog, 
            exog, 
            offset=offset,
            family=sm.families.Poisson()).fit()
    except:
        return((gene, t, 0, 1))
            
    # pv = stats.chi2.sf(-2*(res_fit.llf - fit.llf), df=res_fit.df_resid-fit.df_resid)
    pv = fit.pvalues[t]
    X = exog.values
    pred = fit.predict()
    W = (pred**2 / (   gene_dispersion*(pred + .005749975880258846*pred**2)   ))
    se = np.sqrt(np.diag(np.linalg.pinv(X.T@np.diag(W)@X)))[-1]
    coef = fit.params[t]
    pv = 2*stats.norm.sf(np.abs(coef/se))
    # pv = 0.5
    return((gene, t, coef, pv))


def wald_quasi(
    endog, 
    exog, 
    offset,
    weights=None, 

    gene=None, 
    t=None):
    """
        Perform a Wald ratio test using NB GLM.
    """
    
    try:
        fit = sm.GLM(
            endog, 
            exog, 
            offset=offset,
            family=sm.families.Poisson()).fit()
    except:
        return((gene, t, 0, 1))
            
    # pv = stats.chi2.sf(-2*(res_fit.llf - fit.llf), df=res_fit.df_resid-fit.df_resid)
    pv = fit.pvalues[t]
    X = exog.values
    pred = fit.predict()
    W = (pred**2 / (   gene_dispersion*(pred + sample_dispersion*pred**2)   ))
    se = np.sqrt(np.diag(np.linalg.pinv(X.T@np.diag(W)@X)))[-1]
    coef = fit.params[t]
    pv = 2*stats.norm.sf(np.abs(coef/se))
    # pv = 0.5
    return((gene, t, coef, pv))


def fit_loglinear(endog, exog, offset, gene, t):
    """
        Fit a loglinear model and return the predicted means and model
    """
    
    try:
        fit = sm.Poisson(
            endog, 
            exog, 
            offset=offset).fit(disp=0)
    except:
        fit = sm.GLM(
            endog,
            exog,
            offset=offset,
            family=sm.families.Gaussian(sm.families.links.log())).fit()
        logging.warn(f'fit_loglinear: {gene}, {t} fitted with OLS')
    
    return {
        'gene':gene, 
        't':t,
        'design':exog,
        'pred':fit.predict(),
        'endog':endog,
        'model':fit}
    
    
def quasi_nb_var(mean, scale, dispersion):
    
    return scale*(mean + dispersion*mean**2)


def quasi_nb_objective(scale, dispersion, mean, variance):
    
    valid = (mean > 0) & (variance > 0)
    pred_y = np.log(quasi_nb_var(mean[valid], scale, dispersion))
    y = np.log(variance[valid])
    
    return ((pred_y-y)**2).mean()
    

def fit_quasi_nb(mean, variance):

    dispersion0 = 1
    scale0 = np.median((mean[variance > 0]/variance[variance > 0]))
        
    optim_obj = lambda params: quasi_nb_objective(params[0], params[1], mean, variance)

    res =  minimize(
        optim_obj, 
        [scale0, dispersion0],
        bounds=[(1e-5,None), (1e-10, 10)],
        method='Nelder-Mead',
    )
    return res.x