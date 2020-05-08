import pandas as pd

stocks = pd.read_csv('9.stocks.csv', header='infer')
stocks.index = stocks['Date']
stocks = stocks.drop(['Date'], axis = 1)
print(stocks.head())
