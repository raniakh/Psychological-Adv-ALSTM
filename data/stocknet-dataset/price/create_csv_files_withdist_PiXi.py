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
new_data_path = 'preprocessed_rania_withdist_PiXi/'
start_date = '2013-06-03'
end_date = '2015-12-31'

## Mean holdings
avg_holdings = pd.read_csv('../../Average_holdings_2015.csv')
instruments = pd.read_csv('../../Instruments.csv')
mean_holding_period = round(avg_holdings.days.mean())
print('mean holding period:', mean_holding_period, 'days')
inst_dict = dict(zip(avg_holdings.Symbol, avg_holdings.days))
stockId_to_symbol_map = dict(zip(instruments.InstrumentID, instruments.Symbol))
symbol_to_stockId_map = dict(zip(instruments.Symbol, instruments.InstrumentID))

stocknet_tickers = ['XOM', 'RDS-B', 'PTR', 'CVX', 'TOT', 'BP', 'BHP', 'SNP', 'SLB', 'BBL',
                    'AAPL', 'PG', 'BUD', 'KO', 'PM', 'TM', 'PEP', 'UN', 'UL', 'MO',
                    'JNJ', 'PFE', 'NVS', 'UNH', 'MRK', 'AMGN', 'MDT', 'ABBV', 'SNY', 'CELG',
                    'AMZN', 'BABA', 'WMT', 'CMCSA', 'HD', 'DIS', 'MCD', 'CHTR', 'UPS', 'PCLN',
                    'NEE', 'DUK', 'D', 'SO', 'NGG', 'AEP', 'PCG', 'EXC', 'SRE', 'PPL',
                    'IEP', 'HRG', 'CODI', 'REX', 'SPLP', 'PICO', 'AGFS',
                    'BCH', 'BSAC', 'BRK-A', 'JPM', 'WFC', 'BAC', 'V', 'C', 'HSBC', 'MA',
                    'GE', 'MMM', 'BA', 'HON', 'UTX', 'LMT', 'CAT', 'GD', 'DHR', 'ABB',
                    'GOOG', 'MSFT', 'FB', 'T', 'CHL', 'ORCL', 'TSM', 'VZ', 'INTC', 'CSCO']  # 'GMRE'

print('num of intersection between lists:', len(set(inst_dict.keys()).intersection(stocknet_tickers)))

## create dict for mean holdings for stocknet
stocknet_tickers_dict = {}
for s in stocknet_tickers:
    try:
        stocknet_tickers_dict[s] = int(inst_dict[s])
    except:
        stocknet_tickers_dict[s] = int(mean_holding_period)

distributions = pd.read_csv('../../distributions.csv')
dist_dict = dict([(i, [w1, w2, w3, w4, w5, w6]) for i, w1, w2, w3, w4, w5, w6 in
                  zip(distributions.StockId, distributions.week1, distributions.weeks2,
                      distributions.weeks3, distributions.weeks4, distributions.weeks5,
                      distributions.weeks6)])


def add_gm_features(stock_sample):
    mult_percentage = 100
    stock_sample.loc[:, 'gm_week'] = stock_sample.loc[:, 'Adj Close'].rolling(week).mean()
    stock_sample.loc[:, 'gm_2_weeks'] = stock_sample.loc[:, 'Adj Close'].rolling(weeks2).mean()
    stock_sample.loc[:, 'gm_3_weeks'] = stock_sample.loc[:, 'Adj Close'].rolling(weeks3).mean()
    stock_sample.loc[:, 'gm_4_weeks'] = stock_sample.loc[:, 'Adj Close'].rolling(weeks4).mean()
    stock_sample.loc[:, 'gm_5_weeks'] = stock_sample.loc[:, 'Adj Close'].rolling(weeks5).mean()
    stock_sample.loc[:, 'gm_6_weeks'] = stock_sample.loc[:, 'Adj Close'].rolling(weeks6).mean()

    stock_sample.loc[:, 'gm_week'] = mult_percentage * (
            stock_sample.loc[:, 'gm_week'] / stock_sample.loc[:, 'Adj Close'] - 1)
    stock_sample.loc[:, 'gm_2_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_2_weeks'] / stock_sample.loc[:, 'Adj Close'] - 1)
    stock_sample.loc[:, 'gm_3_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_3_weeks'] / stock_sample.loc[:, 'Adj Close'] - 1)
    stock_sample.loc[:, 'gm_4_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_4_weeks'] / stock_sample.loc[:, 'Adj Close'] - 1)
    stock_sample.loc[:, 'gm_5_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_5_weeks'] / stock_sample.loc[:, 'Adj Close'] - 1)
    stock_sample.loc[:, 'gm_6_weeks'] = mult_percentage * (
            stock_sample.loc[:, 'gm_6_weeks'] / stock_sample.loc[:, 'Adj Close'] - 1)

def add_pr_features(stock_sample):
    stock_sample.loc[:,'pr_week'] = stock_sample.loc[:,'Adj Close'].rolling(week).agg(lambda x: (x>0).mean())
    stock_sample.loc[:,'pr_2_weeks'] = stock_sample.loc[:,'Adj Close'].rolling(weeks2).agg(lambda x: (x>0).mean())
    stock_sample.loc[:,'pr_3_weeks'] = stock_sample.loc[:,'Adj Close'].rolling(weeks3).agg(lambda x: (x>0).mean())
    stock_sample.loc[:,'pr_4_weeks'] = stock_sample.loc[:,'Adj Close'].rolling(weeks4).agg(lambda x: (x>0).mean())
    stock_sample.loc[:,'pr_5_weeks'] = stock_sample.loc[:,'Adj Close'].rolling(weeks5).agg(lambda x: (x>0).mean())
    stock_sample.loc[:,'pr_6_weeks'] = stock_sample.loc[:,'Adj Close'].rolling(weeks6).agg(lambda x: (x>0).mean())

    stock_sample.loc[:,'pr_week'] = MinMaxScaler().fit_transform(stock_sample.loc[:, 'pr_week'].values.reshape(-1,1))
    stock_sample.loc[:,'pr_2_weeks'] = MinMaxScaler().fit_transform(stock_sample.loc[:, 'pr_2_weeks'].values.reshape(-1,1))
    stock_sample.loc[:,'pr_3_weeks'] = MinMaxScaler().fit_transform(stock_sample.loc[:, 'pr_3_weeks'].values.reshape(-1,1))
    stock_sample.loc[:,'pr_4_weeks'] = MinMaxScaler().fit_transform(stock_sample.loc[:, 'pr_4_weeks'].values.reshape(-1,1))
    stock_sample.loc[:,'pr_5_weeks'] = MinMaxScaler().fit_transform(stock_sample.loc[:, 'pr_5_weeks'].values.reshape(-1,1))
    stock_sample.loc[:,'pr_6_weeks'] = MinMaxScaler().fit_transform(stock_sample.loc[:, 'pr_6_weeks'].values.reshape(-1,1))


def add_features(stock):
    stock_path = os.path.join(raw_data_path, '{}'.format(stock))
    stock_sample = pd.read_csv(stock_path)
    add_gm_features(stock_sample)

    #### Adjusting accordingly to the txt ####

    # vector that normalize the prices by adjusted_price(t-1)
    norm_vector = pd.Series(stock_sample['Adj Close']).shift(1).fillna(
        method='bfill')  # fill the first value with the second

    # norm by close at t
    col_adj = ['Open', 'High', 'Low', 'Volume']
    for col in col_adj:
        stock_sample[col] = stock_sample[col] / stock_sample['Close'] - 1

    # norm by 'close' at t-1
    col_adj_t_minus_1 = ['Close', 'Adj Close']
    for col in col_adj_t_minus_1:
        stock_sample[col] = stock_sample[col] / norm_vector - 1

    add_pr_features(stock_sample)

    df = stock_sample.copy()

    # mask between dates
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df.loc[mask]
    stockSymbol = stock.split('.')[0]
    stockId = symbol_to_stockId_map[stockSymbol]
    accountDistribution(df, stockId, stockSymbol)

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
        newfeatures = add_features(stock)
        df_new.drop(df_new.iloc[:, -2:], inplace=True, axis=1)
        df_new = pd.concat([df_new, pd.DataFrame(newfeatures)], axis=1)
        df_new = pd.concat([df_new, pd.DataFrame(tmp_last_columns)], axis=1)
    else:
        tmp_last_columns = df_new.iloc[-num_rows:, -2:].copy()
        # TODO: continue
    return df_new

def accountDistribution(df, stockId, StockSymbol):
    dist = dist_dict[stockId]
    df.loc[:, 'gm_week'] = dist[0]*df['gm_week']
    df.loc[:, 'gm_2_weeks'] = dist[1]*df['gm_2_weeks']
    df.loc[:, 'gm_3_weeks'] = dist[2]*df['gm_3_weeks']
    df.loc[:, 'gm_4_weeks'] = dist[3]*df['gm_4_weeks']
    df.loc[:, 'gm_5_weeks'] = dist[4] * df['gm_5_weeks']
    df.loc[:, 'gm_6_weeks'] = dist[5] * df['gm_6_weeks']

    df.loc[:, 'pr_week'] = dist[0]*df['pr_week']
    df.loc[:, 'pr_2_weeks'] = dist[1]*df['pr_2_weeks']
    df.loc[:, 'pr_3_weeks'] = dist[2]*df['pr_3_weeks']
    df.loc[:, 'pr_4_weeks'] = dist[3]*df['pr_4_weeks']
    df.loc[:, 'pr_5_weeks'] = dist[4] * df['pr_5_weeks']
    df.loc[:, 'pr_6_weeks'] = dist[5] * df['pr_6_weeks']


for stock in os.listdir(original_path):
    try:
        df_new = changeOurppedFiles(stock)
        #df_new = norm_df(df_new)
        # stockSymbol = stock.split('.')[0]
        # stockId = symbol_to_stockId_map[stockSymbol]
        # accountDistribution(df_new, stockId, stockSymbol)
        df_new.to_csv(os.path.join(new_data_path,'{}'.format(stock)),header=None, index=None)
    except Exception as e:
        print(e)
        print('except:', stock)
