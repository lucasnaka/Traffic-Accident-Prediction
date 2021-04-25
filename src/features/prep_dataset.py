# Importing libraries
# Data structures
import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from pyod.models.pca import PCA as PCA_od
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Fix imbalance libs
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.model_selection import train_test_split

# For statistics
from scipy import stats


def dummies(data, columns:list):
    """Convert categorical variables into dummy variables.
    
    Parameters:
        data: DataFrame
        
        columns: list
            Column names in the DataFrame to be encoded.
            
    Returns:
        DataFrame with dummy-coded data
    """
    
    columns = data[columns]
#     dummies = columns.stack().str.get_dummies().max(level=0)
    dummies = pd.get_dummies(columns)
    
    # Join dummies dataset with main dataset
    data = pd.concat([data, dummies], axis = 1)
    
    # Drop columns where get_dummies() was applied
    data = data.drop(columns = columns)
    
    return data

def normalize(data, columns, method='zscore'):
    """Function to normalize numeric features.
    
    Parameters:
        data: DataFrame
        
        columns: list
            List containing column names to be normalized.
        
        method: str, default='zscore'
            Method used to normalize feature.
    
    Returns:
        DataFrame with numeric features normalized.
    """
    
    # Apply the z-score method in Pandas using the .mean() and .std() methods
    if method=='zscore':
        data[columns] = (data[columns] - data[columns].mean()) / data[columns].std()
    
    elif method=='minmax':
        data[columns] = (data[columns] - data[columns].min()) / (data[columns].max() - data[columns].min())
        
    return data

def remove_outliers(data, method='pca'):
    """Function to remove outliers using z-score method or PCA linear dimensionality reduction.
    
    Parameters:
        data: DataFrame
        
        columns: list
            List containing column names to remove outliers.
        
        method: str, default='pca', {'pca', 'zscore'}
            Method used to remove outliers.
    
    Returns:
        DataFrame without outliers.
    """
        
    if method=='pca':
        pca = PCA_od(contamination=0.01, random_state=42)
        X = data.drop(columns=['Target'])
        X = X.select_dtypes(exclude='datetime')
        pca.fit(X)
        pca_predict = pca.predict(X)
        X["pca"] = pca_predict
        outliers = X[X["pca"] == 1].index
        data = data[~data.index.isin(outliers)]

    elif method=='zscore':
        z_score = stats.zscore(data.drop(columns=['Target']))
        abs_z_scores = np.abs(z_score)
        filtered_entries = (abs_z_scores < 3)
        data = data[filtered_entries]
        
    return data

def feature_selection(X_train, X_test, y_train, method='boruta'):
    """Feature selection method.
    
    Parameters:
        X_train: DataFrame
            Array-like of shape (n_samples, n_features).
            
        X_test: DataFrame
            Array-like of shape (n_samples, n_features).
        
        y_train: DataFrame
            Array-like of shape (n_samples, 1).
            
        method: str, default='boruta'
            Feature selection method to be performed.
        
    Returns:
        X_train, X_test, y_train and y_test after the feature selection.
    """
    
    if method=='boruta':
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

        # Define borutapy with model as estimator
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=100)

        # Fit boruta selector to X and y boruta
        boruta_selector.fit(np.array(X_train), np.array(y_train))
        boruta_support = boruta_selector.support_
        boruta_feature = X_train.loc[:,boruta_support].columns.tolist()
    
        X_train = X_train[boruta_feature]
        X_test = X_test[boruta_feature]
    
        return X_train, X_test
    
    return None

def remove_multicollinearity(X_train, X_test, threshold=5.0):
    """Drop features that are highly correlated with each other. 
    Remove columns which vif is greater than threshold parameter.
    
    Parameters:
        X: DataFrame
            Array-like of shape (n_samples, n_features).
            
        threshold: Float
            Value that defines if a column will be removed. If vif is greater than this value, the column will be dropped.
    
    Returns:
        DataFrame without highly correlated features.
    """
    
    variables = list(range(X_train.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X_train.iloc[:, variables].values, ix)\
                for ix in range(X_train.iloc[:, variables].shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > threshold:
            del variables[maxloc]
            dropped = True
    
    return X_train.iloc[:, variables], X_test.iloc[:, variables]
 
def fix_imbalance(X_train,
                  y_train,
                  fix_imbalance_method='SMOTENC',
                  cat_columns=[],
                  random_state=42):
    """Get the imbalance strategy to apply during model training.
    
    Parameters:
        X_train: DataFrame
        
        y_train: Dataframe
            
        fix_imbalance_method: str, default='SMOTENC', possible args={'SMOTENC', 'SMOTE', 'NCL'}
        
        cat_columns: list
            List containing categorical columns. Used in SMOTENC method.
            
        random_state: int, default=42
            Random state for the SMOTE method.
            If int, random_state is the seed used by the random number generator
            If None, the random number generator is the RandomState instance used by np.random.
    
    Returns:
        Imbalance strategy to apply in pipeline.
    """
    
    # Define steps in pipeline
    if fix_imbalance_method=='SMOTENC':        
        cat_col_indices = []
        for cat in cat_columns:
            cat_col_indices = np.append(cat_col_indices, np.where(X_train.columns.str.startswith(cat))[0])
        cat_col_indices = [int(i) for i in cat_col_indices]
        cat_col_indices = sorted(cat_col_indices)

        smote_nc = SMOTENC(categorical_features=cat_col_indices, random_state=random_state, n_jobs=-1)

        X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)
        
        return X_train_resampled, y_train_resampled
            
    elif fix_imbalance_method=='SMOTE':
        return SMOTE(random_state=random_state) # terminar depois

    elif fix_imbalance_method=='undersampling':
        return NeighbourhoodCleaningRule() # terminar depois
    
    else:
        return 'passthrough'