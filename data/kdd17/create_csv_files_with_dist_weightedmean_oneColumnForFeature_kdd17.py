import pandas as pd
import os
from os import listdir
import sys
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

week, weeks2, weeks3, weeks4, weeks5, weeks6 = 7, 14, 21, 28, 35, 42
weeks_list = [week, weeks2, weeks3, weeks4, weeks5, weeks6]
raw_data_path = 'raw/'
original_path = 'ourpped/'
new_data_path = 'preprocessed_rania_withdist_OneColumnForFeature_WeightedMean/'

## Mean holdings
avg_holdings = pd.read_csv('../Average_holdings.csv')
instruments = pd.read_csv('../Instruments.csv')
mean_holding_period = round(avg_holdings.days.mean())
print('mean holding period:', mean_holding_period, 'days')
inst_dict = dict(zip(avg_holdings.Symbol, avg_holdings.days))
stockId_to_symbol_map = dict(zip(instruments.InstrumentID, instruments.Symbol))
symbol_to_stockId_map = dict(zip(instruments.Symbol, instruments.InstrumentID))

KDD_tickers = [f.split('.')[0] for f in listdir(original_path)]
print('num of intersection between lists:', len(set(inst_dict.keys()).intersection(KDD_tickers)))
### create dict for mean holdings for stocknet
KDD_tickers_dict = {}
for s in KDD_tickers:
    try:
        KDD_tickers_dict[s] = int(inst_dict[s])
    except:
        KDD_tickers_dict[s] = int(mean_holding_period)

distributions = pd.read_csv('../distributions.csv')
dist_dict = dict([(i, [w1, w2, w3, w4, w5, w6]) for i, w1, w2, w3, w4, w5, w6 in
                  zip(distributions.StockId, distributions.week1, distributions.weeks2,
                      distributions.weeks3, distributions.weeks4, distributions.weeks5,
                      distributions.weeks6)])


def add_gm_features(stock_sample):  # provide ourpped path
    mult_percentage = 100
    rows_indices = stock_sample.iloc[:, 4] == -123321
    stock_sample.loc[rows_indices, 4] = 0
    #     print(rows_indices)
    #     print(stock_sample.loc[rows_indices, 4])
    stock_sample.loc[:, 'gm_week'] = stock_sample.iloc[:, 4].rolling(week).mean()
    stock_sample.loc[:, 'gm_2_weeks'] = stock_sample.iloc[:, 4].rolling(weeks2).mean()
    stock_sample.loc[:, 'gm_3_weeks'] = stock_sample.iloc[:, 4].rolling(weeks3).mean()
    stock_sample.loc[:, 'gm_4_weeks'] = stock_sample.iloc[:, 4].rolling(weeks4).mean()
    stock_sample.loc[:, 'gm_5_weeks'] = stock_sample.iloc[:, 4].rolling(weeks5).mean()
    stock_sample.loc[:, 'gm_6_weeks'] = stock_sample.iloc[:, 4].rolling(weeks6).mean()

    stock_sample.loc[:, 'gm_week'] = mult_percentage * (
            stock_sample.loc[:, 'gm_week'] / stock_sample.iloc[:, 4] - 1)
    stock_sample.loc[:, 'gm_2_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_2_weeks'] / stock_sample.iloc[:, 4] - 1)
    stock_sample.loc[:, 'gm_3_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_3_weeks'] / stock_sample.iloc[:, 4] - 1)
    stock_sample.loc[:, 'gm_4_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_4_weeks'] / stock_sample.iloc[:, 4] - 1)
    stock_sample.loc[:, 'gm_5_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_5_weeks'] / stock_sample.iloc[:, 4] - 1)
    stock_sample.loc[:, 'gm_6_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_6_weeks'] / stock_sample.iloc[:, 4] - 1)


def add_pr_features(stock_sample):  # provide ourpped path
    rows_indices = stock_sample.iloc[:, 4] == -123321
    stock_sample.loc[rows_indices, 4] = 0
    stock_sample.loc[:, 'pr_week'] = stock_sample.iloc[:, 4].rolling(week).agg(lambda x: (x > 0).mean())
    stock_sample.loc[:, 'pr_2_weeks'] = stock_sample.iloc[:, 4].rolling(weeks2).agg(lambda x: (x > 0).mean())
    stock_sample.loc[:, 'pr_3_weeks'] = stock_sample.iloc[:, 4].rolling(weeks3).agg(lambda x: (x > 0).mean())
    stock_sample.loc[:, 'pr_4_weeks'] = stock_sample.iloc[:, 4].rolling(weeks4).agg(lambda x: (x > 0).mean())
    stock_sample.loc[:, 'pr_5_weeks'] = stock_sample.iloc[:, 4].rolling(weeks5).agg(lambda x: (x > 0).mean())
    stock_sample.loc[:, 'pr_6_weeks'] = stock_sample.iloc[:, 4].rolling(weeks6).agg(lambda x: (x > 0).mean())

    stock_sample.loc[:, 'pr_week'] = MinMaxScaler().fit_transform(stock_sample.loc[:, 'pr_week'].values.reshape(-1, 1))
    stock_sample.loc[:, 'pr_2_weeks'] = MinMaxScaler().fit_transform(
        stock_sample.loc[:, 'pr_2_weeks'].values.reshape(-1, 1))
    stock_sample.loc[:, 'pr_3_weeks'] = MinMaxScaler().fit_transform(
        stock_sample.loc[:, 'pr_3_weeks'].values.reshape(-1, 1))
    stock_sample.loc[:, 'pr_4_weeks'] = MinMaxScaler().fit_transform(
        stock_sample.loc[:, 'pr_4_weeks'].values.reshape(-1, 1))
    stock_sample.loc[:, 'pr_5_weeks'] = MinMaxScaler().fit_transform(
        stock_sample.loc[:, 'pr_5_weeks'].values.reshape(-1, 1))
    stock_sample.loc[:, 'pr_6_weeks'] = MinMaxScaler().fit_transform(
        stock_sample.loc[:, 'pr_6_weeks'].values.reshape(-1, 1))


def add_features(stock):
    #     stock_path = os.path.join(original_path, '{}'.format(stock))
    #     stock_sample = pd.read_csv(stock_path)
    #     add_gm_features(stock_sample)
    #     add_pr_features(stock_sample)
    add_gm_features(stock)
    add_pr_features(stock)

    #     df = stock_sample.copy()
    df = stock.copy()

    return np.vstack([df['gm_week'].values, df['gm_2_weeks'].values,
                      df['gm_3_weeks'].values, df['gm_4_weeks'].values,
                      df['gm_5_weeks'].values, df['gm_6_weeks'].values,
                      df['pr_week'].values, df['pr_2_weeks'].values,
                      df['pr_3_weeks'].values, df['pr_4_weeks'].values,
                      df['pr_5_weeks'].values, df['pr_6_weeks'].values]).T


def changeOurppedFiles(stock, num_rows=0):
    # original df
    stock_path = os.path.join(original_path, '{}'.format(stock))
    df_orig = pd.read_csv(stock_path, header=None)
    df_new = df_orig.copy()
    # insert new features
    if num_rows == 0:
        tmp_last_columns = df_new.iloc[:, -2:].copy()
        df_new.drop(df_new.iloc[:, -2:], inplace=True, axis=1)
        newfeatures = add_features(df_new)
        #         newfeatures  = add_features(stock)
        #         df_new = pd.concat([df_new, pd.DataFrame(newfeatures)], axis=1)
        df_new = pd.concat([df_new, pd.DataFrame(tmp_last_columns)], axis=1)
    else:
        tmp_last_columns = df_new.iloc[-num_rows:, -2:].copy()
        # TODO: continue
    return df_new


def fillMissingData(df):
    df.replace(' ', np.nan, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(skipna=True, numeric_only=True), inplace=True)
    rows_indices = df.iloc[:, 1] == -123321
    df.loc[rows_indices, :] = -123321


def accountDistribution(df, stockId, StockSymbol):
    dist = dist_dict[stockId]
    weighted_gmavg = dist[0]*df['gm_week'] + dist[1]*df['gm_2_weeks'] + dist[2]*df['gm_3_weeks']\
                     + dist[3]*df['gm_4_weeks'] + dist[4] * df['gm_5_weeks'] + dist[5] * df['gm_6_weeks']
    weighted_pravg = dist[0]*df['pr_week'] + dist[1]*df['pr_2_weeks'] + dist[2]*df['pr_3_weeks']\
                     + dist[3]*df['pr_4_weeks'] + dist[4] * df['pr_5_weeks'] + dist[5] * df['pr_6_weeks']
    df.loc[:, 'gm_weightedMean'] = weighted_gmavg
    df.loc[:, 'pr_weightedMean'] = weighted_pravg

    df.drop(columns=['gm_week', 'gm_2_weeks', 'gm_3_weeks', 'gm_4_weeks',
                     'gm_5_weeks', 'gm_6_weeks', 'pr_weeks', 'pr_2_weeks',
                     'pr_3_weeks', 'pr_4_weeks', 'pr_5_weeks', 'pr_6_weeks'],
            axis=1, inplace=True)


for stock in os.listdir(original_path):
    try:
        df_new = changeOurppedFiles(stock)
        # df_new = norm_df(df_new)
        fillMissingData(df_new)
        stockSymbol = stock.split('.')[0]
        stockId = symbol_to_stockId_map[stockSymbol]
        accountDistribution(df_new, stockId, stockSymbol)
        df_new.to_csv(os.path.join(new_data_path, '{}'.format(stock)), header=None, index=None)
    except Exception as e:
        print(e)
        print('except:', stock)
