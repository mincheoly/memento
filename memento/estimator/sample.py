import logging

import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats


def sample_mean(X: sparse.csc_matrix, size_factor: np.array):
    """ Computes the normalized sample mean. """
    
    row_weight = (1/size_factor).reshape([1, -1])
    return sparse.csc_matrix.dot(row_weight, X).ravel()/X.shape[0]


def sample_variance(X: sparse.csc_matrix, size_factor: np.array):
    """ Computes the normalized sample variance. """
    
    """ Compute the variances. """

    n_obs = X.shape[0]
    row_weight = (1 / size_factor).reshape([1, -1])
    row_weight_sq = (1 / size_factor ** 2).reshape([1, -1])

    mm_M1 = sparse.csc_matrix.dot(row_weight, X).ravel() / n_obs
    mm_M2 = sparse.csc_matrix.dot(row_weight_sq, X.power(2)).ravel() / n_obs

    mean = mm_M1
    variance = (mm_M2 - mm_M1 ** 2)

    return variance if len(variance) > 1 else float(variance)