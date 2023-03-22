import pandas as pd
import os
from os import listdir
import sys
import yaml
import numpy as np
from sklearn import preprocessing

kdd_path = '../kdd17/ourpped/'
stocknet_path = '../stocknet-dataset/price/ourpped'
distributions_file_path = '../distributions.csv'
instruments = pd.read_csv('../Instruments.csv')

stockId_to_symbol_map = dict(zip(instruments.InstrumentID, instruments.Symbol))
symbol_to_stockId_map = dict(zip(instruments.Symbol, instruments.InstrumentID))

kdd_tickers = [f.split('.')[0] for f in listdir(kdd_path)]
stocknet_tickers = [f.split('.')[0] for f in listdir(stocknet_path)]
tickers_union = list(set(kdd_tickers + stocknet_tickers))

distributions = pd.read_csv(distributions_file_path)
dist_dict = dict([(i, [w1, w2, w3, w4, w5, w6]) for i, w1, w2, w3, w4, w5, w6 in
                  zip(distributions.StockId, distributions.week1, distributions.weeks2,
                      distributions.weeks3, distributions.weeks4, distributions.weeks5,
                      distributions.weeks6)])
distributions_kdd_and_stocknet = {}

for stock in tickers_union:
    stockSymbol = stock.split('.')[0]
    try:
        stockId = symbol_to_stockId_map[stockSymbol]
        if stockId in dist_dict:
            distributions_kdd_and_stocknet[stockId] = dist_dict[stockId]
    except Exception as e:
        print(e)
        print('except:', stock)


print(len(distributions_kdd_and_stocknet.keys()))
# calculate average of distributions of all stocks. normalize.
week1 = np.mean([v[0] for v in distributions_kdd_and_stocknet.values()])
week2 = np.mean([v[1] for v in distributions_kdd_and_stocknet.values()])
week3 = np.mean([v[2] for v in distributions_kdd_and_stocknet.values()])
week4 = np.mean([v[3] for v in distributions_kdd_and_stocknet.values()])
week5 = np.mean([v[4] for v in distributions_kdd_and_stocknet.values()])
week6 = np.mean([v[5] for v in distributions_kdd_and_stocknet.values()])
weeks_sum = week1+week2+week3+week4+week5
weeks_list = [week1, week2, week3, week4, week5, week6]
normalized_weeks_dist = (weeks_list + min(weeks_list)) / sum(weeks_list + min(weeks_list))
print('.....')
df = pd.DataFrame(normalized_weeks_dist)
df.to_csv(os.path.join('../synthetic/', '{}'.format('dist_for_synthetic.csv')), header=None, index=None)