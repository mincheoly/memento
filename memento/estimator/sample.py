import logging

import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import util as util


def mean(X: sparse.csc_matrix, size_factor: np.array):
    
    """ Computes the normalized sample mean. """
    
    row_weight = (1/size_factor).reshape([1, -1])
    return sparse.csc_matrix.dot(row_weight, data).ravel()/n_obs