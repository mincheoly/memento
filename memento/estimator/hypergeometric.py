import logging

import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats

from memento.estimator.sample import sample_variance


def return_float(val):
    
    return val if len(val) > 1 else float(val)


def fill_invalid(val, group_name):
    """ Fill invalid entries by randomly selecting a valid entry. """

    # negatives and nan values are invalid values for our purposes
    invalid_mask = np.less_equal(val, 0., where=~np.isnan(val)) | np.isnan(val)
    num_invalid = invalid_mask.sum()

    if num_invalid == val.shape[0]:
        # if all values are invalid, there are no valid values to choose from, so return all nans
        logging.warning(f"all bootstrap variances are invalid for group {group_name}")
        return np.full(shape=val.shape, fill_value=np.nan)

    val[invalid_mask] = np.random.choice(val[~invalid_mask], num_invalid)
    
    return val


def unique_expr(expr, size_factor):
    """
        Find (approximately) unique combinations of expression values and size factors.
        The random component is for mapping (expr, size_factor) to a single number.
        This can certainly be performed more efficiently using sparsity.
    """

    code = expr.dot(np.random.random(expr.shape[1]))
    approx_sf = size_factor

    code += np.random.random() * approx_sf

    _, index, count = np.unique(code, return_index=True, return_counts=True)

    expr_to_return = expr[index].toarray()

    return (
        1 / approx_sf[index].reshape(-1, 1), 
        1 / approx_sf[index].reshape(-1, 1) ** 2, 
        expr_to_return, 
        count)


def hg_mean(X: sparse.csc_matrix, size_factor: np.array):
    """ Inverse variance weighted mean. """

    mean = (X.sum(axis=0).A1+1)/(size_factor.sum()+1)
    
    return return_float(mean)


def bootstrap_mean_for_gene(
        unique_expr: np.array,
        bootstrap_freq: np.array,
        q: float,
        n_obs: int,
        inverse_size_factor: np.array,
        inverse_size_factor_sq: np.array
):
    """ Compute the bootstrapped variances for a single gene expression frequencies."""

    means = ((unique_expr * bootstrap_freq).sum(axis=0)+1) / ((bootstrap_freq/inverse_size_factor).sum(axis=0)+1)

    return means


def hg_sem_for_gene(
        X: sparse.csc_matrix,
        q: float,
        approx_size_factor: np.array,
        num_boot: int = 5000,
        group_name: tuple = (),
        return_boot_samples=False,
):
    """ Compute the standard error of the variance for a SINGLE gene. """
    
    n_obs = X.shape[0]
    inv_sf, inv_sf_sq, expr, counts = unique_expr(X, approx_size_factor)

    gen = np.random.Generator(np.random.PCG64(5))
    gene_rvs = gen.multinomial(n_obs, counts / counts.sum(), size=num_boot).T

    mean = bootstrap_mean_for_gene(
        unique_expr=expr,
        bootstrap_freq=gene_rvs,
        n_obs=n_obs,
        q=q,
        inverse_size_factor=inv_sf,
        inverse_size_factor_sq=inv_sf_sq
    )
    
    if return_boot_samples:
        return mean

    sem = np.nanstd(mean)
    selm = np.nanstd(np.log(mean))
    sel1pm = np.nanstd(np.log(mean+1))
    
    return sem, selm, sel1pm


def hg_variance(X: sparse.csc_matrix, q: float, size_factor: np.array, group_name=None):
    """ Compute the variances. """

    n_obs = X.shape[0]
    row_weight = (1 / size_factor).reshape([1, -1])
    row_weight_sq = (1 / size_factor ** 2).reshape([1, -1])

    mm_M1 = sparse.csc_matrix.dot(row_weight, X).ravel() / n_obs
    mm_M2 = sparse.csc_matrix.dot(row_weight_sq, X.power(2)).ravel() / n_obs - (1 - q) * sparse.csc_matrix.dot(
        row_weight_sq, X).ravel() / n_obs

    mean = mm_M1
    variance = (mm_M2 - mm_M1 ** 2)

    return return_float(variance)


def bootstrap_variance_for_gene(
        unique_expr: np.array,
        bootstrap_freq: np.array,
        q: float,
        n_obs: int,
        inverse_size_factor: np.array,
        inverse_size_factor_sq: np.array
):
    """ Compute the bootstrapped variances for a single gene expression frequencies."""

    mm_M1 = (unique_expr * bootstrap_freq).sum(axis=0) / (bootstrap_freq/inverse_size_factor).sum(axis=0)
    mm_M2 = (unique_expr ** 2 * bootstrap_freq * inverse_size_factor_sq - (
                1 - q) * unique_expr * bootstrap_freq * inverse_size_factor_sq).sum(axis=0) / n_obs

    variance = mm_M2 - mm_M1 ** 2
    return mm_M1, variance


def hg_sev_for_gene(
        X: sparse.csc_matrix,
        q: float,
        approx_size_factor: np.array,
        mv_fit: np.array,
        num_boot: int = 5000,
        group_name: tuple = (),
        return_boot_samples=False,
):
    """ Compute the standard error of the variance for a SINGLE gene. """
    
    if X.max() < 2:
        return np.nan, np.nan
    
    n_obs = X.shape[0]
    inv_sf, inv_sf_sq, expr, counts = unique_expr(X, approx_size_factor)

    gen = np.random.Generator(np.random.PCG64(5))
    gene_rvs = gen.multinomial(n_obs, counts / counts.sum(), size=num_boot).T

    mean, var = bootstrap_variance_for_gene(
        unique_expr=expr,
        bootstrap_freq=gene_rvs,
        n_obs=n_obs,
        q=q,
        inverse_size_factor=inv_sf,
        inverse_size_factor_sq=inv_sf_sq
    )

    mean, var = fill_invalid(var, group_name)
    res_var = residual_variance(mean, var, mv_fit)
    
    if return_boot_samples:
        return var

    sev = np.nanstd(var)
    selv = np.nanstd(np.log(var))
    serv = np.nanstd(res_var)
    selrv = np.nanstd(np.log(res_var))

    return sev, selv, serv, selrv


def fit_mv_regressor(mean, var):
    """
        Perform regression of the variance against the mean.
    """

    cond = (mean > 0) & (var > 0)
    m, v = np.log(mean[cond]), np.log(var[cond])

    poly = np.polyfit(m, v, 2)
    return poly
    f = np.poly1d(z)
    

def residual_variance(mean, var, mv_fit):

    cond = (mean > 0) & (var > 0)
    rv = np.zeros(mean.shape)*np.nan

    f = np.poly1d(mv_fit)
    with np.errstate(invalid='ignore'):
        rv[cond] = np.exp(np.log(var[cond]) - f(np.log(mean[cond])))
    return rv
