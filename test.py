import os
import pandas as pd
from src.predictor import Predicter
from src.utils import blockPrint,enablePrint


class ValuateTesting():
    def __init__(self,path_csv,del_module=False,use_meta=True):
        '''
        Args:
            path_csv (string): path to test.csv
        Return:
            performance
        '''
        self.df_test = pd.read_csv(path_csv)
