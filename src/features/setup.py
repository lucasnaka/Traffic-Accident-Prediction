# Importing libraries
# Data structures
import pandas as pd
import numpy as np

from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))

from sklearn.model_selection import train_test_split

# Local libs
from features import prep_dataset as prep

class PrepareDataset(object):
    def __init__(self, 
                 dataset,
                 test_date_range:tuple=('2020-01-01','2020-12-31'),
                ) -> None:

        ###################################
        #           parameters
        ###################################
        self.dataset = dataset
        self.test_date_min  = test_date_range[0]
        self.test_date_max  = test_date_range[1]
      
    def setup(self,
              target_variable:str='Target',
              train_size:float=0.7,
              categorical_features:list=[],
              numerical_features:list=[],
              indices:list=[],
              remove_outliers:bool=False,
              remove_outliers_method:str='pca',
              normalize:bool=False,
              normalize_method:str='zscore',
              feature_selection:bool=False,
              feature_selection_method:str='boruta',
              remove_multicollinearity:bool=False,
              fix_imbalance:bool=False,
              fix_imbalance_method:str='SMOTENC',
              dummies:list=[],
             ) -> None:
        """Prepare dataset to input in the training step.
        
        Parameters:
            target_variable: str, default='Target'
                Target column name in the training dataframe.
                
            *train_size: float, default=0.7
                Proportion of the dataset to be used for training and validation. Should be between 0.0 and 1.0.
                
            categorical_features: list, default=[]
                List containing categorical features.
                
            numerical_features: list, default=[]
                List containing numerical features.
                
            *normalize: bool, default=False
                When set to True, it transforms the numeric features by scaling them to a given range. Type of scaling is defined by the normalize_method parameter.
                
            *normalize_method: str, default='zscore'
                Defines the method for scaling. By default, normalize method is set to 'zscore' The standard zscore is calculated as z = (x - u) / s. Ignored when normalize is not True. The other options are:
                minmax: scales and translates each feature individually such that it is in the range of 0 - 1.
                
            indices: list,
                List with the columns names that will be set as indexes.
            
            remove_outliers: bool, default=False
                When set to True, PCA method is applied by default to remove outliers.
                
            remove_outliers_method: str, default='pca'
                Method to be used to remove outliers.
            
            feature_selection: bool, default=False
                If True performs the feature selection.
                
            feature_selection_method: str, default='boruta'
                Feature selection method to be performed.
            
            remove_multicollinearity: bool, default=False
                If True performs the remove multicollinearity.
            
            *fix_imbalance: bool, default=False
                When set to True, SMOTENC (Synthetic Minority Over-sampling Technique Nominal Continuous) is applied by default to create synthetic datapoints for minority class.
                
            fix_imbalance_method: str, default='SMOTENC'
                Imbalance method to be performed.
                
            dummies: list
                List of column names to perform dummies.
        """
        
        # Get dummies
        if dummies:
            self.dataset = prep.dummies(self.dataset, dummies)
        
        # Defining indexes for the dataset
        if indices:
            indices = indices
            self.dataset = self.dataset.set_index(indices)
        
        # Removing outliers
        if remove_outliers:
            self.dataset = prep.remove_outliers(self.dataset, remove_outliers_method)
        
        # Normalizing numeric features
        if normalize:
            self.dataset = prep.normalize(self.dataset, numerical_features, 'zscore')
        
        self.dataset_test = self.dataset.loc[(self.dataset['data_inversa'] >= self.test_date_min)
                                            & (self.dataset['data_inversa'] <= self.test_date_max)]
        
        del self.dataset_test['data_inversa']
        del self.dataset['data_inversa']
        
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.dataset.drop(['Target'], axis=1), self.dataset['Target'], train_size=train_size)
        
        # Fix imbalance
        if fix_imbalance:
            self.X_train, self.y_train = prep.fix_imbalance(X_train=self.X_train,
                                                            y_train=self.y_train,
                                                            fix_imbalance_method='SMOTENC',
                                                            cat_columns=categorical_features,
                                                            random_state=42)
        
        self.dataset_train = self.X_train
        self.dataset_train['Target'] = self.y_train
        self.dataset_validation = self.X_validation
        self.dataset_validation['Target'] = self.y_validation
        
#         # Feature selection
#         if feature_selection:
#             self.X_train, self.X_test = prep.feature_selection(self.X_train, self.X_test, self.y_train, method=feature_selection_method)
        
#         # Remove multicollinearity
#         if remove_multicollinearity:
#             self.X_train, self.X_test = prep.remove_multicollinearity(self.X_train, self.X_test, threshold=5.0)