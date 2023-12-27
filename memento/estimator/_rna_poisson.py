import logging
import pandas as pd
import scipy.sparse as sparse
import numpy as np 

from ._base import EstimatorBase

class RNAPoisson(EstimatorBase):
    """
    Class for holding functions that are used across multiple estimators.
    """
    
    
    def mean(self, data: sparse.csr_matrix, size_factor: np.array):
        """
        Implements the Good-Turing estimator at the pseudobulk level.
        """

        row_weight = (1/size_factor).reshape([1, -1])
        mean = sparse.csc_matrix.dot(row_weight, data).ravel()/n_obs
        
        return mean
    
    
    def variance(self, data: sparse.csr_matrix, size_factor: np.array):
        
        n_obs = data.shape[0]
        row_weight = (1 / size_factor).reshape([1, -1])
        row_weight_sq = (1 / size_factor ** 2).reshape([1, -1])

        mm_M1 = sparse.csc_matrix.dot(row_weight, data).ravel() / n_obs
        mm_M2 = sparse.csc_matrix.dot(row_weight_sq, data.power(2)).ravel() / n_obs - sparse.csc_matrix.dot(
            row_weight_sq, data).ravel() / n_obs

        mean = mm_M1
        variance = (mm_M2 - mm_M1 ** 2)

        return variance
    
    
    def covariance(self, data: sparse.csr_matrix, size_factor: np.array):
        
        pass
    
    
    def bootstrap_mean(self, values: np.array, freq: np.array, inv_sf: np.array, inv_sf_sq: np.array):
        """
            Computes the mean from bootstrapped values and frequencies. 
            
            :values: expression values
            :freq: frequencies of the :values:
            :inv_sf: inverse of cell sizes
            :inv_sf_sq: inverse of cell sizes squared
        """
        
        pass

    
    def bootstrap_mean(
        self, 
        values: np.array, 
        freq: np.array, 
        inv_sf: np.array, 
        inv_sf_sq: np.array):
        """
            Computes the variance from bootstrapped values and frequencies. 
            
            :values: expression values
            :freq: frequencies of the :values:
            :inv_sf: inverse of cell sizes
            :inv_sf_sq: inverse of cell sizes squared
        """
        
        pass