import logging
import os
import pandas as pd 


class MementoBase():
    """
    Class for holding functions that are used across multiple memento models.
    """
    
    def save_estimates(self, path):
        
        if len(self.estimates) == 0:
            
            logging.info('save_estimates: Empty estimates, nothing was saved')
        
        if os.path.exists(path):
            logging.info('save_estimates: folder already exists, possibly overwriting')
        else:
            os.mkdir(path)
            
        for est, df in self.estimates.items():
            
            outfile = os.path.join(path, f'{est}.csv')
            df.to_csv(outfile)
            
            
    def load_estimates(self, path):
        
        file_names = [f for f in os.listdir(path) if '.csv' in f]
        for file in file_names:
            
            infile = os.path.join(path, file)
            estimator_name = file.split('.csv')[0]
            self.estimates[estimator_name] = pd.read_csv(infile, index_col=0)