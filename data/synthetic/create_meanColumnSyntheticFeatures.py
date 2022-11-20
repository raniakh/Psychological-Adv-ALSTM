import pandas as pd
import os
from os import listdir
import sys
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

raw_data_path = 'raw/'
original_path = 'preprocessed/'
new_data_path = 'meanColumn/'


def changeOurppedFiles(stock):
    stock_path = os.path.join(original_path, '{}'.format(stock))
    df_orig = pd.read_csv(stock_path, header=None)
    df_new = df_orig.copy()
    tmp_last_columns = df_new.iloc[:, -2:].copy()
    df_new.drop(df_new.iloc[:, -2:], inplace=True, axis=1)
    gm_mean = np.mean((df_new.iloc[:, 10], df_new.iloc[:, 11], df_new.iloc[:, 12], df_new.iloc[:, 13],
                      df_new.iloc[:, 14], df_new.iloc[:, 15]), axis=0)
    pr_mean = np.mean((df_new.iloc[:, 16], df_new.iloc[:, 17], df_new.iloc[:, 18], df_new.iloc[:, 19],
                       df_new.iloc[:, 20], df_new.iloc[:, 21]), axis=0)
    df_new.drop(df_new.iloc[:, 10:], inplace=True, axis=1)
    df_return = pd.concat([df_new, pd.DataFrame(gm_mean), pd.DataFrame(pr_mean), pd.DataFrame(tmp_last_columns)], axis=1)
    return df_return


for stock in os.listdir(original_path):
    try:
        df_new = changeOurppedFiles(stock)
        df_new.to_csv(os.path.join(new_data_path, '{}'.format(stock)), header=None, index=None)
    except Exception as e:
        print(e)
        print('except:', stock)