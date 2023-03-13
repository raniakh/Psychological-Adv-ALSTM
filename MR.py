import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef
from pandas import Series, DataFrame

data_path = './data/stocknet-dataset/price/ourpped'
def load_prices(stock, seq=10, date_format='%Y-%m-%d'):
    print(stock)
    df = DataFrame()
    stock_file = np.genfromtxt(os.path.join(data_path, stock), dtype=float, delimiter=',', skip_header=False)
    df['price'] = stock_file[:, 4]
    df['volume'] = stock_file[:, -1]
    df['label'] = stock_file[:, -2]
    df['price_moving_avg'] = df['price'].rolling(window=30).mean()
    df['vol_moving_avg'] = df['volume'].rolling(window=30).mean()

    df = df[df['price_moving_avg'].notna()]
    prices = df['price']
    pmavgplot = df['price_moving_avg']
    vmavgplot = df['vol_moving_avg']
    prices.plot(label=stock, legend=True)
    pmavgplot.plot(label='mavg 30d', legend=True)
    vmavgplot.plot(secondary_y=True, label='volume avg 30d', legend=True)
    plt.title(stock)
    plt.show()
    df['price lower than mavg'] = df['price_moving_avg'].gt(df['price'])
    df['volume higher than vmavg'] = df['vol_moving_avg'].gt(df['volume'])
    z=1
    PL=0.00
    start_price = df['price'].head(1)
    start_price = float(start_price)
    print('start price: ', start_price)
    end_price = df['price'].tail(1)
    end_price = float(end_price)
    print('end price: ', end_price)
    if start_price ==0:
        return 0, 0
    Return = (PL/start_price)
    Return_Per = ":.2%".format(Return)
    output = []
    truelabel = []
    for index, row in df.iterrows():
        if row['volume higher than vmavg'] == 1:
            if(row['price lower than mavg']) == 1:
                if z==1:
                    print(index, row['price'], '-BUY')
                    close_adj = row['price']
                    PL = PL - close_adj
                    z=z-1
                    output.append(1)
                    truelabel.append(row['label'])
        else:
            if row['volume higher than vmavg'] == 0:
                if row['price lower than mavg'] == 0:
                    if z==0:
                        print(index, row['price'], '-SELL')
                        close_adj = row['price']
                        PL = PL + close_adj
                        Return = (PL/start_price)
                        Return_Per = "{:.2%}".format(Return)
                        print("Total profit/Loss $", round(PL, 2))
                        print("Total Return %", Return, "\n")
                        z = z+1
                        output.append(0)
                        truelabel.append(row['label'])

    acc = accuracy_score(truelabel, output)
    mcc = matthews_corrcoef(truelabel, output)
    return acc, mcc


if __name__ == '__main__':
    total_perf = 0
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    acc_score, mcc_score = 0, 0
    acc_array, mcc_array = [],[]
    for stock in fnames:
        acc, mcc = load_prices(stock)
        acc_score += acc
        mcc_score += mcc
        acc_array.append(acc)
        mcc_array.append(mcc)



    print('manual acc_score ',acc_score/len(fnames))
    print('manual mcc_score', mcc_score/len(fnames))
    print('median acc_array ',np.median(acc_array))
    print('median mcc array ',np.median(mcc_array))
    print('average acc_array ', np.average(acc_array))
    print('averag mcc_array ',np.average(mcc_array))
    print('mcc_array ', mcc_array)