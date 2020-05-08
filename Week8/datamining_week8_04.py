import pandas as pd

stocks = pd.read_csv('9.stocks.csv', header='infer')
stocks.index = stocks['Date']
stocks = stocks.drop(['Date'], axis = 1)
stocks.head()

import numpy as np 

N, d = stocks.shape
delta = pd.DataFrame(100*np.divide(stocks.iloc[1:,:].values-stocks.iloc[:N-1,:].values,
        stocks.iloc[:N-1,:].values), columns=stocks.columns, index=stocks.iloc[1:].index)
delta.head()

meanValue = delta.mean()
covValue = delta.cov()
print(meanValue)
print(covValue)
