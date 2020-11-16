from sklearn.cluster import KMeans
import pandas as pd

def order_cluster(cluster_field_name, target_field_name, df, ascending):
    """Order `cluster_field_name` by `target_field_name`.
    If `ascending=True`, ascending order will be considered, else descending.
     

    Args:
        cluster_field_name ([type]): [description]
        target_field_name ([type]): [description]
        df ([type]): [description]
        ascending ([type]): [description]

    Returns:
        [type]: [description]
    """    
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

def clustering_dimension(df, dimension, clusterting_order_column, cluster_size, cluster_name, ascending=True):
    clusterting_order_column_list = sorted(list(df[clusterting_order_column].unique()))
    for i in clusterting_order_column_list:
        X = df[df[clusterting_order_column]<=i][dimension].values.reshape(-1, 1)
        df.loc[df[clusterting_order_column]==i, cluster_name] = KMeans(cluster_size).fit_predict(X) if len(X) >= cluster_size else np.array([1]*len(X))
        df.loc[df[clusterting_order_column]==i, cluster_name] = order_cluster(cluster_name, dimension,df[df[clusterting_order_column]==i], ascending)[cluster_name].values
