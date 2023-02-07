"""

"""
import os
import pandas as pd
import numpy as np
from constants import DATA_DIR, TARGET, ASD, TD
import constants
import threading
from sklearn.model_selection import train_test_split

lock = threading.Lock()


class DataReaderFunc:
    def __init__(self, data: pd.DataFrame=None, phenofile: pd.DataFrame=None,
                 behavioral: str=None):
        self.df = self._handle_data_input(data)

        self.df_pheno = self._handle_pheno_input(phenofile)
        if 'subj_id' in self.df_pheno.columns:
            self.df_pheno.set_index('subj_id', inplace=True)

        self.df_total = None
        self.df_train_ = None
        self.df_test_ = None

        # self._get_behavioral_columns()

    def _handle_pheno_input(self, pheno):
        """
        Handling pheno as a parameter passed to the constructor
        :return:
        """
        if pheno is None:
            with lock:
                df = pd.read_csv(constants.FF_DATA_PHENO)
        elif isinstance(pheno, pd.DataFrame):
                df = pheno
        elif isinstance(pheno, str):
            with lock:
                df = pd.read_csv(pheno)
        else:
            raise TypeError(f'pheno can be either None or dataframe')
        df.loc[df['DX_GROUP'] == 2, 'DX_GROUP'] = 0
        return df[['FILE_ID', "DX_GROUP", "AGE_AT_SCAN", "SEX"]]

    def _handle_data_input(self, data):
        """
        Handling data as a parameter passed to the constructor
        :return:
        """
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, str):
            with lock:
                df = pd.read_csv(data, index_col=0)
            
        return df

    def _get_behavioral_columns(self):
        for col in self.df_pheno.columns:
            if self.report_type.lower() in col.lower():
                self.behavioral_columns_.append(col)

    def _merge_pheno_data(self):
        postfix = "_".join(self.df.index[0].split('_')[-2:])
        df_temp = self.df.reset_index()
        self.df_pheno['FILE_ID'] = self.df_pheno['FILE_ID'].apply(lambda x: f"{x}_{postfix}")
        self.df_total = pd.merge(df_temp, self.df_pheno, left_on="index", right_on="FILE_ID", how="inner")
        return self.df_total

    def set_params(self, **params) -> None:
        for key, val in params.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise KeyError(f'{key} is not a valid parameter')

    def run(self):
        self._merge_pheno_data()
        self.df_total.drop("FILE_ID", axis=1, inplace=True)
        self.df_total.set_index("index", inplace=True)
        df_train, df_test = train_test_split(self.df_total, test_size=0.2)
        return self.df_total, df_train, df_test


if __name__ == '__main__':
    # df = pd.read_csv(DATA_DIR['medianMmedianP'], index_col=0)
    # df_p = pd.read_csv(DATA_DIR['pheno'], index_col='subj_id')

    # divisor = DataDivisor(df, df_p, 'srs')
    # df = divisor.divide()
    # divisor = DataDivisor()
    # df = divisor.get_group('comm','sever')
    # x = 0

    list_input = [os.path.join(constants.FF_DATA_PATH, "filt_global_rois_aal/_CORR.csv")]
    for inp in list_input:
        dr = DataReaderFunc(inp, constants.FF_DATA_PHENO)
        df = dr.run()
