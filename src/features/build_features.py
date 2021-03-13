# Importing libraries
# Data structures
import pandas as pd
import numpy as np

# OS library
from os import path
import sys

# For statistics
from scipy import stats
import swifter

# For clustering
from sklearn.cluster import KMeans

# For querying
import pandasql as ps

# For plots
import plotly.express as px

sys.path.append(path.join(path.dirname(__file__), '..'))

class BuildFeatures(object):
    """Build features and merge all datasets into a dataframe ready to be trained.
    
    Parameters:
    
    train_date_range: tuple, default=('2017-01-01','2019-12-31')
        Train date range to be splited.
        
    test_date_range: tuple, default=('2020-01-01','2020-12-31')
        Test date range to be splited.
        
    Examples
    --------
    """
    def __init__(self,
                 dataset:object,
                 train_date_range:tuple=('2017-01-01','2019-12-31'),
                 test_date_range:tuple=('2020-01-01','2020-12-31')):

        ###################################
        #           Parameters
        ###################################
        self.train_date_min = train_date_range[0]
        self.train_date_max = train_date_range[1]
        self.test_date_min  = test_date_range[0]
        self.test_date_max  = test_date_range[1]
        self.datatran       = dataset.datatran
        
    def clustering_kmeans_method(self, n_cluster_method='elbow'):
        if n_cluster_method=='elbow':
            def calculate_wcss(data, n_max=10):
                """Calculate within cluster sum-of-square.

                Parameters:

                data: dataframe 
                    Data to be fitted in K-means.

                n_max: int, default=10
                    Max number of clusters to test.

                Returns:

                wcss - List containing within clusters sum-of-squares.
                """
                wcss = []
                for n in range(2, n_max):
                    kmeans = KMeans(n_clusters=n)
                    kmeans.fit(X=data)
                    wcss.append(kmeans.inertia_)

                return wcss

            def optimal_number_of_clusters(wcss):
                """Determine optimal number of clusters by calculating the distance between a point and a line.

                Parameters:
                
                    wcss: list
                        List containing within clusters sum-of-squares.

                Returns:

                    n_optimal: int
                        Optimal number of clusters.
                """
                x1, y1 = 2, wcss[0]
                x2, y2 = 10, wcss[len(wcss)-1]

                distances = []
                for i in range(len(wcss)):
                    x0 = i+2
                    y0 = wcss[i]
                    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
                    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                    distances.append(numerator/denominator)

                n_optimal = distances.index(max(distances)) + 2

                return n_optimal

            df_kmeans = self.dataset[['latitude', 'longitude', 'risco_morte']]

            # Calculating the within clusters sum-of-squares for n_max cluster amounts
            sum_of_squares = calculate_wcss(df_kmeans)

            # Calculating the optimal number of clusters
            n_optimal = optimal_number_of_clusters(sum_of_squares)

            # Running kmeans to our optimal number of clusters
            kmeans = KMeans(n_clusters=n_optimal, random_state=42)
            df_kmeans['cluster_coords'] = kmeans.fit_predict(df_kmeans)

            df_kmeans['cluster_coords'].value_counts().plot(kind='bar', title='Qtde amostras por cluster')

            return df_kmeans
        return None

    def weekday_process(self, weekday_column):
        """Rename weekday column values.

        Parameters:
            
        weekday_column: str
            Name of weekday column.

        Returns:
        
        Return dataset with weekday column values renamed.
        """
        self.dataset[weekday_column] = np.select(
            [
                self.dataset[weekday_column].str.contains('seg'),
                self.dataset[weekday_column].str.contains('ter'),
                self.dataset[weekday_column].str.contains('qua'),
                self.dataset[weekday_column].str.contains('qui'),
                self.dataset[weekday_column].str.contains('sex'),
                self.dataset[weekday_column].str.contains('sab'),
                self.dataset[weekday_column].str.contains('dom')
            ],
            [
                'seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom'
            ],
            ''
        )

    def create_risk_feature(self, days_to_analyse=365):            
        self.dataset['valor_1'] = 1

        # Criar dataset com qtd de acidentes por br/km num periodo de 1 ano ate a data do acidente em questao
        df_acidentes_brkm = self.dataset.groupby(['br', 'km']).rolling(f'{days_to_analyse}D', on="data_inversa")['valor_1'].sum().reset_index(name='qtd_acidentes_brkm')
        df_acidentes_brkm.drop_duplicates(subset=['br', 'km', 'data_inversa'], keep='last', inplace=True)

        # Criar dataset com qtd total de acidentes num periodo de 1 ano ate a data do acidente em questao
        df_acidentes_brasil = self.dataset.groupby(['valor_1']).rolling(f'{days_to_analyse}D', on="data_inversa")['valor_1'].sum().reset_index(name='qtd_acidentes_brasil')
        df_acidentes_brasil = df_acidentes_brasil.drop_duplicates(subset='data_inversa', keep='last')[['data_inversa', 'qtd_acidentes_brasil']]

        # Join datasets
        self.dataset = self.dataset.merge(df_acidentes_brkm, how='left', on=['br', 'km', 'data_inversa'])
        self.dataset = self.dataset.merge(df_acidentes_brasil, how='left', on=['data_inversa'])

        # Criar atributo risco
        self.dataset['risco'] = self.dataset['qtd_acidentes_brkm']/self.dataset['qtd_acidentes_brasil']

        # Remover colunas indesejadas
        del self.dataset['qtd_acidentes_brkm']
        del self.dataset['qtd_acidentes_brasil']
        del self.dataset['valor_1']

    def create_fatal_risk_feature(self, days_to_analyse=365):        
        self.dataset['contem_vitima_fatal'] = self.dataset.apply(lambda x: 1 if x['mortos'] != 0 else 0, axis=1)
        dataset_mortes = self.dataset[self.dataset['mortos']!=0]

        # Criar dataset com qtd de acidentes por br/km num periodo de 1 ano ate a data do acidente em questao
        df_acidentes_brkm = dataset_mortes.groupby(['br', 'km']).rolling(f'{days_to_analyse}D', on="data_inversa", closed='left')['contem_vitima_fatal'].sum().reset_index(name='qtd_acidentes_brkm')
        df_acidentes_brkm.drop_duplicates(subset=['br', 'km', 'data_inversa'], keep='last', inplace=True)

        # Criar dataset com qtd total de acidentes num periodo de 1 ano ate a data do acidente em questao
        df_acidentes_brasil = dataset_mortes.groupby(['contem_vitima_fatal']).rolling(f'{days_to_analyse}D', on="data_inversa", closed='left')['contem_vitima_fatal'].sum().reset_index(name='qtd_acidentes_brasil')
        df_acidentes_brasil = df_acidentes_brasil.drop_duplicates(subset='data_inversa', keep='last')[['data_inversa', 'qtd_acidentes_brasil']]

        # Join datasets
        self.dataset = self.dataset.merge(df_acidentes_brkm, how='left', on=['br', 'km', 'data_inversa'])
        self.dataset = self.dataset.merge(df_acidentes_brasil, how='left', on=['data_inversa'])

        # Criar atributo risco
        self.dataset['risco_morte'] = self.dataset['qtd_acidentes_brkm']/self.dataset['qtd_acidentes_brasil']
        self.dataset['risco_morte'].fillna(0, inplace=True)

        # Remover colunas indesejadas
        del self.dataset['qtd_acidentes_brkm']
        del self.dataset['qtd_acidentes_brasil']

    def accident_in_holiday_window(self, data_inicio_analise, data_fim_analise, days_offset_holiday=2):
        df_holidays = pd.read_parquet('../data/raw/holidays.parquet')

        df_holidays["data"] = pd.to_datetime(df_holidays["data"], format='%d/%m/%Y')
        df_holidays = df_holidays[(df_holidays['data'] >= data_inicio_analise) & (df_holidays['data'] <= data_fim_analise)]
        df_holidays = df_holidays.add_prefix('holiday_')

        df_holidays.fillna('', inplace=True)
        df_holidays[['holiday_municipio']] = df_holidays[['holiday_municipio']].apply(
            lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.
            decode('utf-8').str.replace('[^\w\s]', '').str.lower().str.strip())

        df_holidays[f'holiday_date_minus_{days_offset_holiday}'] = df_holidays['holiday_data'] + pd.DateOffset(-days_offset_holiday)
        df_holidays[f'holiday_date_plus_{days_offset_holiday}'] = df_holidays['holiday_data'] + pd.DateOffset(days_offset_holiday)

        df_holidays = df_holidays[(df_holidays['holiday_tipo'] == 'ESTADUAL') |
                                  (df_holidays['holiday_tipo'] == 'NACIONAL') |
                                  (df_holidays['holiday_nome'] == 'Carnaval')]

        dataset = self.dataset
        
        sqlcode = f'''
        select *
        from dataset
        inner join df_holidays on 
            dataset.data_inversa >= df_holidays.holiday_date_minus_{days_offset_holiday}
            AND dataset.data_inversa <= df_holidays.holiday_date_plus_{days_offset_holiday}
        WHERE
            -- (df_holidays.holiday_tipo LIKE 'MUNICIPAL' AND dataset.municipio = df_holidays.holiday_municipio)
            (df_holidays.holiday_tipo LIKE 'ESTADUAL' AND dataset.uf = df_holidays.holiday_uf)
            OR df_holidays.holiday_tipo LIKE 'NACIONAL'
            OR df_holidays.holiday_nome LIKE 'Carnaval'
        '''

        dataset_near_holidays = ps.sqldf(sqlcode,locals())

        dataset_near_holidays["data_inversa"] = pd.to_datetime(dataset_near_holidays["data_inversa"])
        dataset_near_holidays["holiday_data"] = pd.to_datetime(dataset_near_holidays["holiday_data"])
        dataset_near_holidays['diff_ac_feriado_dias'] = dataset_near_holidays['data_inversa'] - dataset_near_holidays['holiday_data']
        dataset_near_holidays['diff_ac_feriado_dias'] = dataset_near_holidays['diff_ac_feriado_dias'].astype('timedelta64[D]').astype(int)

        priority_list = [0]
        if days_offset_holiday > 0:
            for days in range(1, days_offset_holiday+1):
                priority_list.extend([-days, days])
        priority_list.append(np.inf)
        dataset_near_holidays['diff_ac_feriado_dias'] = dataset_near_holidays['diff_ac_feriado_dias'].astype('category')
        dataset_near_holidays['diff_ac_feriado_dias'] = pd.Categorical(dataset_near_holidays['diff_ac_feriado_dias'], categories=priority_list, ordered=True)
        dataset_near_holidays = dataset_near_holidays.sort_values('diff_ac_feriado_dias').groupby('id', as_index=False).first()
        dataset_near_holidays['em_janela_feriado'] = 1

        self.dataset = self.dataset.merge(dataset_near_holidays[['id', 'em_janela_feriado']], how='left', on=['id'])

        self.dataset['em_janela_feriado'] = self.dataset['em_janela_feriado'].fillna(0)
        self.dataset.drop_duplicates(inplace=True)

        self.dataset.sort_values(["data_inversa"], ascending=True, inplace=True)
    
    def build_features(self):
        self.dataset = self.datatran
        
        min_data = min(self.dataset['data_inversa'])

        self.dataset['d'] = (self.dataset['data_inversa'] - min_data)
        self.dataset['d'] / pd.Timedelta(1, unit='d')
        self.dataset['d'] = self.dataset['d'].astype('timedelta64[D]')+1

        self.dataset.sort_values(by='d', ascending=True, inplace=True)
        
        self.create_risk_feature(days_to_analyse=365)
        self.create_fatal_risk_feature(days_to_analyse=365)
        print('Fatal risk calculated')
        
        # Remover dados de acidentes do primeiro ano de analise por nao haver dados do ano antecedente para criar o atributo risco (lembrar que a analise é feita em um periodo de 1 ano)
        self.dataset = self.dataset.loc[self.dataset['data_inversa'] >= self.train_date_min, :]

        # Transformar dia da semana em dado categórico numérico
        self.weekday_process(weekday_column='dia_semana')

        # Transformar coordenadas (latitude, longitude) em espaço cartesiano
        self.dataset['coordenada_x'] = np.cos(self.dataset['latitude']) * np.cos(self.dataset['longitude'])
        self.dataset['coordenada_y'] = np.cos(self.dataset['latitude']) * np.sin(self.dataset['longitude'])
        self.dataset['coordenada_z'] = np.sin(self.dataset['latitude'])

        # Criar atributo que indica distância em dias entre data do acidente e data de feriados
        # Considerar feriados apenas entre a data de analise

        self.accident_in_holiday_window(data_inicio_analise=self.train_date_min, 
                                        data_fim_analise=self.train_date_max, 
                                        days_offset_holiday=2)
        print('Accident in holiday window calculated')

        # Remoção de outliers
        z_score = stats.zscore(self.dataset['pessoas'])
        abs_z_scores = np.abs(z_score)
        filtered_entries = (abs_z_scores < 3)
        #USAR A LINHA DE BAIXO SE AUMENTAR QTD DE COLUNAS NO ZSCORE
        #filtered_entries = (abs_z_scores < 3).all(axis=1)
        self.dataset = self.dataset[filtered_entries]
        print('Ouliers removed')

        # Clustering using K-means method
        df_kmeans = self.clustering_kmeans_method(n_cluster_method='elbow')
        self.dataset = self.dataset.merge(df_kmeans[['latitude', 'longitude', 'risco_morte', 'cluster_coords']], how='left', on=['latitude','longitude','risco_morte'])
        self.dataset.drop_duplicates(subset='id', inplace=True)
        fig = px.scatter(self.dataset, x="longitude", y="latitude", color="cluster_coords", title='Cluster de coordenadas', width=500, height=500)
        fig.show()

        # Drop unwanted columns
        drop_columns = [
            'horario'
            , 'br'
            , 'km'
            , 'municipio'
            , 'causa_acidente'
            , 'tipo_acidente'
            , 'classificacao_acidente'
            , 'ano'
            , 'mortos'
            , 'feridos_leves'
            , 'feridos_graves'
            , 'ilesos'
            , 'ignorados'
            , 'feridos'
            , 'veiculos'
            , 'latitude'
            , 'longitude'
            , 'regional'
            , 'delegacia'
            , 'uop'
            , 'd'
        ]

        self.dataset.drop(drop_columns, axis=1, inplace=True)