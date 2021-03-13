# Importing libraries
# Data structures
import pandas as pd
import numpy as np

def prep_datatran(dataset, initial_date, final_date):
    dataset['data_inversa'] = pd.to_datetime(dataset['data_inversa'])
    dataset['ano'] = dataset['data_inversa'].dt.year
    
    dataset = dataset.loc[(dataset['data_inversa'] >= initial_date) & (dataset['data_inversa'] <= final_date), :]
    
    dataset['latitude'] = dataset['latitude'].astype('float64')
    dataset['longitude'] = dataset['longitude'].astype('float64')
    
    dataset.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    # String normalization 
    string_columns = ['dia_semana', 
                      'municipio',
                      'causa_acidente',
                      'tipo_acidente',
                      'classificacao_acidente',
                      'fase_dia',
                      'sentido_via',
                      'condicao_metereologica',
                      'tipo_pista',
                      'tracado_via',
                      'uso_solo',
                      'regional',
                      'delegacia',
                      'uop']

    dataset[string_columns] = dataset[string_columns].apply(
        lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.
        decode('utf-8').str.replace('[^\w\s]', '').str.lower().str.strip())
    
    dataset['uso_solo'] = dataset.apply(lambda x: 'urbano' if x['uso_solo'] == 'sim' else 'rural', axis=1)
    
    dataset.loc[dataset["condicao_metereologica"] == 'ignorada',"condicao_metereologica"] = 'ignorado'
    dataset = dataset[dataset["condicao_metereologica"]!='ignorado']

    dataset.loc[dataset["condicao_metereologica"] == 'cau claro',"condicao_metereologica"] = 'ceu claro'
    
    # Fill nan
    dataset['km'] = dataset['km'].apply(lambda x: str(x).replace(',','.'))
    dataset = dataset[~(dataset['br']=='(null)')]
    dataset['br'] = dataset['br'].where(pd.notnull(dataset['br']), None)
    dataset[['br','km']] = dataset[['br','km']].fillna(dataset[['br','km']].mode().iloc[0])
    dataset['br'] = dataset['br'].astype(int)
    dataset['km'] = dataset['km'].astype(float).fillna(dataset['km'].mode().iloc[0]).astype(float).astype(int)
    
    dataset[string_columns] = dataset[string_columns].replace('null', np.nan)
    dataset[string_columns] = dataset[string_columns].fillna(dataset[string_columns].mode().iloc[0])
    
    return dataset