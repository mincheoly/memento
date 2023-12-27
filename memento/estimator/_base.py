import logging
import pandas as pd
import numpy as np
import scipy.sparse as sparse

class EstimatorBase(): # Base class lining out the functions of the estimator class.
    """
    Class for holding functions that are used across multiple estimators.
    """
    
    
    def mean(self, data: sparse.csr_matrix, size_factor: np.array):
        
        
        pass
    
    
    def variance(self, data: sparse.csr_matrix, size_factor: np.array):
        
        pass
    
    
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

    
    def bootstrap_variance(
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
    
    
    def bootstrap_covariance(
        self, 
        values1: np.array, 
        values2: np.array,
        freq: np.array, 
        inv_sf: np.array, 
        inv_sf_sq: np.array):
        """
            Computes the covariance from bootstrapped values and frequencies. 
            
            :values: expression values
            :freq: frequencies of the :values:
            :inv_sf: inverse of cell sizes
            :inv_sf_sq: inverse of cell sizes squared
        """
    
        pass


    def fill_invalid(self, val, group_name):
        """ Fill invalid entries by randomly selecting a valid entry. """

        # negatives and nan values are invalid values for our purposes
        invalid_mask = np.less_equal(val, 0., where=~np.isnan(val)) | np.isnan(val)
        num_invalid = invalid_mask.sum()

        if num_invalid == val.shape[0]:
            # if all values are invalid, there are no valid values to choose from, so return all nans
            logging.info(f"all bootstrap variances are invalid for group {group_name}")
            return True, np.full(shape=val.shape, fill_value=np.nan)

        val[invalid_mask] = np.random.choice(val[~invalid_mask], num_invalid)
    
        return False, val


    def unique_expr(self, expr, size_factor):
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
    
    def return_float(self, val):

        return val if len(val) > 1 else float(val)


    def fill_invalid(self, val, group_name):
        """ Fill invalid entries by randomly selecting a valid entry. """

        # negatives and nan values are invalid values for our purposes
        invalid_mask = np.less_equal(val, 0., where=~np.isnan(val)) | np.isnan(val)
        num_invalid = invalid_mask.sum()

        if num_invalid == val.shape[0]:
            # if all values are invalid, there are no valid values to choose from, so return all nans
            logging.info(f"all bootstrap variances are invalid for group {group_name}")
            return True, np.full(shape=val.shape, fill_value=np.nan)

        val[invalid_mask] = np.random.choice(val[~invalid_mask], num_invalid)

        return False, val