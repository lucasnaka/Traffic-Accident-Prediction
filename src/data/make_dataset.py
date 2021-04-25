# Importing libraries
# Data structures
import numpy as np
import pandas as pd
from datetime import datetime

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
                 verbose:bool=False,
                 path_files='../data'):

        ###################################
        #           Parameters
        ###################################
        self.train_date_min = train_date_range[0]
        self.train_date_max = train_date_range[1]
        self.test_date_min  = test_date_range[0]
        self.test_date_max  = test_date_range[1]
        self.verbose        = verbose
        self.path_files     = path_files
                 
    def get_datatran(self):
        """Get a dataframe with the "datatran" dataset.
        """
#         dt_min = datetime.strptime(self.train_date_min, '%Y-%m-%d')
#         dt_max = datetime.strptime(self.test_date_max, '%Y-%m-%d')
        
#         datatran_list = []
        
#         for year in range(dt_min.year, dt_max.year+1):
#             df_datatran = pd.read_csv(f"{self.path_files}/raw/datatran{year}.csv", delimiter=";", encoding='iso-8859-1')
#             datatran_list.append(df_datatran)
        
#         self.datatran = pd.concat(datatran_list)

        self.datatran = pd.read_csv(f"{self.path_files}/processed/df_datatran_2009_2020.csv", delimiter=",", encoding='iso-8859-1')
        
        # Prepare dataset
        self.datatran = pp.prep_datatran(self.datatran, initial_date=self.train_date_min, final_date=self.test_date_max, verbose=self.verbose)
        
    def get_acidentes(self):
        """Get a dataframe with the "acidentes" dataset.
        """
        dt_min = datetime.strptime(self.train_date_min, '%Y-%m-%d')
        dt_max = datetime.strptime(self.test_date_max, '%Y-%m-%d')
        
        acidentes_list = []
        
        for year in range(dt_min.year, dt_max.year+1):
            df_acidentes = pd.read_csv(f"{self.path_files}/raw/acidentes{year}.csv", delimiter=";", encoding='iso-8859-1')
            acidentes_list.append(df_acidentes)
        
        self.acidentes = pd.concat(acidentes_list)
        
        # Prepare dataset
        self.acidentes = pp.prep_acidentes(self.acidentes, initial_date=self.train_date_min, final_date=self.test_date_max)
    
    def get_volume_praca(self):
        self.vol_praca = pd.read_csv(f"{self.path_files}/raw/volume_praca_2021.csv", delimiter=";", encoding='iso-8859-1')
    
    def get_placa_ultrapassagem(self):
        self.placa_ultrapassagem = pd.read_csv(f"{self.path_files}/raw/proibido_ultrapassar_03_2021.csv", delimiter=";", encoding='iso-8859-1')
    
    def get_placa_velocidade(self):
        self.placa_velocidade1 = pd.read_csv(f"{self.path_files}/raw/velocidade_maxima_08_2020.csv", delimiter=";", encoding='iso-8859-1')
        self.placa_velocidade2 = pd.read_csv(f"{self.path_files}/raw/velocidade_maxima_09_2020.csv", delimiter=";", encoding='iso-8859-1')
        self.placa_velocidade3 = pd.read_csv(f"{self.path_files}/raw/velocidade_maxima_10_2020.csv", delimiter=";", encoding='iso-8859-1')
        self.placa_velocidade4 = pd.read_csv(f"{self.path_files}/raw/velocidade_maxima_11_2020.csv", delimiter=";", encoding='iso-8859-1')
        self.placa_velocidade5 = pd.read_csv(f"{self.path_files}/raw/velocidade_maxima_12_2020.csv", delimiter=";", encoding='iso-8859-1')
        self.placa_velocidade6 = pd.read_csv(f"{self.path_files}/raw/velocidade_maxima_01_2021.csv", delimiter=";", encoding='iso-8859-1')
        self.placa_velocidade7 = pd.read_csv(f"{self.path_files}/raw/velocidade_maxima_02_2021.csv", delimiter=";", encoding='iso-8859-1')
        self.placa_velocidade8 = pd.read_csv(f"{self.path_files}/raw/velocidade_maxima_03_2021.csv", delimiter=";", encoding='iso-8859-1')
    
        self.placa_velocidade = pd.concat([self.placa_velocidade1
            ,self.placa_velocidade2
            ,self.placa_velocidade3
            ,self.placa_velocidade4
            ,self.placa_velocidade5
            ,self.placa_velocidade6
            ,self.placa_velocidade7
            ,self.placa_velocidade8])
    
    def make(self):
        """Methodology to make dataset.
        """
        self.get_datatran()
        self.get_acidentes()
#         self.get_volume_praca()
#         self.get_placa_ultrapassagem()
#         self.get_placa_velocidade()

if __name__ == '__main__':
    
    # Running the MAKE function
    make()