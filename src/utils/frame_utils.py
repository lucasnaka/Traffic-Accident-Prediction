from scipy import stats
import numpy as np
import pandas as pd

def stratified_df(df, column, length=1000):
    """Generate a stratified pandas.DataFrame of length `length`,
    keeping the same ratio to each of unique `column` value.

    Args:
        df ([type]): [description]
        column ([type]): [description]
        length (int, optional): [description]. Defaults to 1000.

    Returns:
        [type]: [description]
    """    
    return df.groupby(column, group_keys=False).apply(lambda x: x.sample(min(len(x), length)))

def date_time_features(n, df, id_column, date_column):
    """Generate features from `date_column` to get the `n` last purchases, differentiate,
    generate statistics measures (mean, std, median, mode) and the next purchase date interval

    Args:
        n ([type]): [description]
        df ([type]): [description]
        id_column ([type]): [description]
        date_column ([type]): [description]
    """    
    df.sort_values([id_column, date_column], inplace=True)
    for i in range(1, n+2):
        df['date_minus_'+str(i)] = df.groupby(id_column)[date_column].shift(i)
        df['date_diff_minus_'+str(i)] = (df[date_column] - df['date_minus_'+str(i)]).dt.days  
    df[['date_diff_minus_'+str(i) for i in range(1, n+1)]] = df[['date_diff_minus_'+str(i) for i in range(1, n+1)]].diff(axis=1)
    df.drop(columns=['date_diff_minus_1', 'date_minus_'+str(n+1)], axis=1, inplace=True)
    df.rename(columns=dict(zip([i for i in df.columns if i.startswith('date_diff_minus_')], ['date_diff_minus_'+str(i+1) for i in range(n)])), inplace=True)
    df['last_purchase_moving_average'] = df.groupby(id_column)['date_diff_minus_1'].transform(lambda x: x.ewm(span=n).mean()).values
    df['next_purchase_date'] = df.groupby(id_column)[date_column].shift(-1)
    df['next_purchase_in_days'] = (df['next_purchase_date'] - df[date_column] ).dt.days
    df['Next Purchase Revenue with Discounts'] = df.groupby(id_column)['Revenue with discounts'].shift(-1)
    # PyCaret Notebook Time Series
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year
    df['dayofyear'] = df[date_column].dt.dayofyear
    df['dayofmonth'] = df[date_column].dt.day
    df['weekofyear'] = df[date_column].dt.weekofyear
    df['flag_covid'] = pd.Series(np.where(df[date_column] >= np.datetime64('2020-03-03'), 1, 0), index=df.index) #flag for COVID-19
    # df['rolling_mean_7_next_purchase'] = df['next_purchase_in_days'].shift(7).rolling(window=7).mean()
    # df['lag_7_next_purchase'] = df['next_purchase_in_days'].shift(7)
    # df['lag_15_next_purchase'] = df['next_purchase_in_days'].shift(15)
    # df['lag_last_3_months_next_purchase'] = df['next_purchase_in_days'].shift(12).rolling(window=15).mean()
    
    df['last_purchases_mean'] = np.nanmean(df[[i for i in df.columns if i.startswith('date_diff_minus_')]], axis=1)
    df['last_purchases_std'] = np.nanstd(df[[i for i in df.columns if i.startswith('date_diff_minus_')]], axis=1)
    df['last_purchases_median'] = np.nanmedian(df[[i for i in df.columns if i.startswith('date_diff_minus_')]], axis=1)
    df['last_purchases_mode'] = stats.mode(df[[i for i in df.columns if i.startswith('date_diff_minus_')]], axis=1, nan_policy='omit')[0]
    
    df.sort_values([id_column, date_column], inplace=True)
    
    return df


def last_n_features(n, df, id_column, date_column, feature):
    """Generate features from n last interactions, differentiate,
    generate statistics measures (mean, std, median, mode) and the next target interval

    Args:
        n ([type]): [description]
        df ([type]): [description]
        id_column ([type]): [description]
        date_column ([type]): [description]
        feature ([type]): [description]
    """    
    df.sort_values([id_column, date_column], inplace=True)
    for i in range(1, n+1):
        df[f'{feature} (n - {i})'] = df.groupby(id_column)[feature].shift(i)
        df[f'{feature} (n - {i}) diff 1'] = df.groupby(id_column)[feature].shift(i)
        df[f'current - {feature} (n - {i})'] = (df[feature] - df[f'{feature} (n - {i})'])
        
    df[['date_diff_minus_'+str(i) for i in range(1, n+1)]] = df[['date_diff_minus_'+str(i) for i in range(1, n+1)]].diff(axis=1)
    df.drop(columns=['date_diff_minus_1', 'date_minus_'+str(n+1)], axis=1, inplace=True)
    df.rename(columns=dict(zip([i for i in df.columns if i.startswith('date_diff_minus_')], ['date_diff_minus_'+str(i+1) for i in range(n)])), inplace=True)
    df['last_purchase_moving_average'] = df.groupby(id_column)['date_diff_minus_1'].transform(lambda x: x.ewm(span=n).mean()).values
    df['next_purchase_date'] = df.groupby(id_column)[date_column].shift(-1)
    df['next_purchase_in_days'] = (df['next_purchase_date'] - df[date_column] ).dt.days
    df['Next Purchase Revenue with Discounts'] = df.groupby(id_column)['Revenue with discounts'].shift(-1)
    # PyCaret Notebook Time Series
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year
    df['dayofyear'] = df[date_column].dt.dayofyear
    df['dayofmonth'] = df[date_column].dt.day
    df['weekofyear'] = df[date_column].dt.weekofyear
    df['flag_covid'] = pd.Series(np.where(df[date_column] >= np.datetime64('2020-03-03'), 1, 0), index=df.index) #flag for COVID-19
    # df['rolling_mean_7_next_purchase'] = df['next_purchase_in_days'].shift(7).rolling(window=7).mean()
    # df['lag_7_next_purchase'] = df['next_purchase_in_days'].shift(7)
    # df['lag_15_next_purchase'] = df['next_purchase_in_days'].shift(15)
    # df['lag_last_3_months_next_purchase'] = df['next_purchase_in_days'].shift(12).rolling(window=15).mean()
    
    df['last_purchases_mean'] = np.nanmean(df[[i for i in df.columns if i.startswith('date_diff_minus_')]], axis=1)
    df['last_purchases_std'] = np.nanstd(df[[i for i in df.columns if i.startswith('date_diff_minus_')]], axis=1)
    df['last_purchases_median'] = np.nanmedian(df[[i for i in df.columns if i.startswith('date_diff_minus_')]], axis=1)
    df['last_purchases_mode'] = stats.mode(df[[i for i in df.columns if i.startswith('date_diff_minus_')]], axis=1, nan_policy='omit')[0]
    
    df.sort_values([id_column, date_column], inplace=True)
    
    return df
 
    
def reduce_mem_usage(df, verbose=True):
    """Reduce memory usage of a dataframe, by changing the type of data.

    Args:
        df ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df