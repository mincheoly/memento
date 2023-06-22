import logging

import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats


def sample_mean(X: sparse.csc_matrix, size_factor: np.array):
    
    """ Computes the normalized sample mean. """
    
    row_weight = (1/size_factor).reshape([1, -1])
    return sparse.csc_matrix.dot(row_weight, X).ravel()/X.shape[0]