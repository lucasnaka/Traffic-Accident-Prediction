# Importing libraries
# Data structures
import numpy as np
import pandas as pd

# OS library
import sys
from os import path

# In-House library
from src.data import preprocessing as pp

sys.path.append(path.join(path.dirname(__file__), '..'))

class MakeDataset(object):
    """Get the datasets and perform feature engineering.
    
    Parameters:
                 
    train_date_range: tuple, default=('2017-01-01','2019-12-31')
        Train date range to be splited.
        
    test_date_range: tuple, default=('2020-01-01','2020-12-31')
        Test date range to be splited.
                 
    path_files: str, default='../data/'
        Path to the local data files
        
    Examples
    --------
    from src.data import make_dataset as md
    
    # Calling MakeDataset class and parsing parameters
    dataset = md.MakeDataset(train_date_range=('2017-01-01','2019-12-31')
                            ,test_date_range=('2020-01-01','2020-12-31'))

    # Running the make method to download all the datasets
    dataset.make()
    """
    def __init__(self,
                 train_date_range:tuple=('2017-01-01','2019-12-31'),
                 test_date_range:tuple=('2020-01-01','2020-12-31'),
                 path_files='../data/'):

        ###################################
        #           Parameters
        ###################################
        self.train_date_min = train_date_range[0]
        self.train_date_max = train_date_range[1]
        self.test_date_min  = test_date_range[0]
        self.test_date_max  = test_date_range[1]
        self.path_files     = path_files
                 
    def get_datatran(self):
        """Get a dataframe with the "datatran" dataset.
        """
        self.datatran = pd.read_csv(f"{self.path_files}/processed/df_datatran_2009_2020.csv", delimiter=",", encoding='iso-8859-1')
        
        # Prepare dataset
        self.datatran = pp.prep_datatran(self.datatran, initial_date=self.train_date_min, final_date=self.test_date_max)
                 
    def make(self):
        """Methodology to make dataset.
        """
        self.get_datatran()

if __name__ == '__main__':
    
    # Running the MAKE function
    make()