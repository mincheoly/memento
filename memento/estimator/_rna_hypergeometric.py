import logging
import pandas as pd
import scipy.sparse as sparse
import numpy as np 

from ._base import EstimatorBase

class RNAHypergeometric(EstimatorBase):
    """
    Class for holding functions that are used across multiple estimators.
    """
    
    
    def __init__(self, capture_efficiency):
        
        self.q = capture_efficiency
    
    
    def mean(self, data: sparse.csr_matrix, size_factor: np.array):
        """
        Implements the Good-Turing estimator at the pseudobulk level.
        """

        pb = data.sum(axis=0).A1.astype(int)

        pb_freqs = pd.Series(pb).value_counts().sort_index()

        r = pb_freqs.index.values
        nr = pb_freqs.values

        z = pd.Series(pb_freqs.index).rolling(window=3, center=True).apply(lambda x: x.iloc[-1]-x.iloc[0]).values
        z[0] = 2
        z[-1] = z[-2]
        z = 2*nr/z

        r_star = (r+1)*np.concatenate([z[1:], z[-1:]])/z

        pb_star = pb.copy().astype(float)
        for r_val in r[r < 5]:

            pb_star[pb == r_val] = r_star[np.where(r==r_val)[0]][0]
            
        return pb_star/pb.sum()
        
    
    def variance(self, data: sparse.csr_matrix, size_factor: np.array):
        
        n_obs = data.shape[0]
        row_weight = (1 / size_factor).reshape([1, -1])
        row_weight_sq = (1 / size_factor ** 2).reshape([1, -1])

        mm_M1 = sparse.csc_matrix.dot(row_weight, data).ravel() / n_obs
        mm_M2 = sparse.csc_matrix.dot(row_weight_sq, data.power(2)).ravel() / n_obs - (1 - self.q) * sparse.csc_matrix.dot(
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