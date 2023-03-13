import pandas as pd
import os
from os import listdir
import sys
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

original_path = 'base/'
new_data_path = 'psycho/'
mean_holding_days = 15 # =7*0.46368034570523+14*0.290266332400527+21*0.0360918678865935+28*0.0168704867465785+35*0.188624557440505+42*0.00446640982056413
mult_percentage = 100

def changeOurppedFiles(stock):
    stock_path = os.path.join(original_path, '{}'.format(stock))
    df_orig = pd.read_csv(stock_path, header=None)
    df_new = df_orig.copy()
    tmp_last_columns = df_new.iloc[:, -2:].copy()
    df_new.drop(df_new.iloc[:, -2:], inplace=True, axis=1)

    gm = df_new.iloc[:, 4].rolling(mean_holding_days).mean()
    gm = mult_percentage * (gm / df_new.iloc[:, 4] - 1)

    pr = df_new.iloc[:, 4].rolling(mean_holding_days).agg(lambda x: (x > 0).mean())
    pr = np.array(pr)
    pr = MinMaxScaler().fit_transform(pr.reshape(-1, 1))

    df_return = pd.concat([df_new, pd.DataFrame(gm), pd.DataFrame(pr), pd.DataFrame(tmp_last_columns)], axis=1)
    return df_return


for stock in os.listdir(original_path):
    try:
        df_new = changeOurppedFiles(stock)
        df_new.dropna(axis=0, inplace=True)
        df_new.to_csv(os.path.join(new_data_path, '{}'.format(stock)), header=None, index=None)
    except Exception as e:
        print(e)
        print('except:', stock)