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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy.linalg import inv

X = delta.as_matrix()
S = covValue.as_matrix()
for i in range(3):
    X[:,i] = X[:,i]-meanValue[i]

def mahalanobis(row):
    return np.matmul(row,S).dot(row)

anomaly_score = np.apply_along_axis(mahalanobis, axis = 1, arr =X)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(delta.MSFT, delta.F, delta.BAC, c = anomaly_score, cmap ='jet')
ax.set_xlabel('Microsoft')
ax.set_ylabel('Ford')
ax.set_zlabel('Bank of America')
fig.colorbar(p)

anom = pd.DataFrame(anomaly_score, index = delta.index, columns=['Anomaly score'])
result = pd.concat((delta,anom), axis = 1)
result.nlargest(2,'Anomaly score')

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,6))

ts = delta[440:447]
ts.plot.line(ax=ax1)
ax1.set_xticks(range(7))
ax1.set_xticklabels(ts.index)
ax1.set_ylabel('Percent Change')

ts = delta[568:575]
ts.plot.line(ax=ax2)
ax2.set_xticks(range(7))
ax2.set_xticklabels(ts.index)
ax2.set_ylabel('Percent Change')

from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial import distance
knn = 4
nbrs = NearestNeighbors(n_neighbors=knn, metric=distance.euclidean).fit(delta.as_matrix())
distances, indices = nbrs.kneighbors(delta.as_matrix())
anomaly_score = distances[:,knn-1]

anom = pd.DataFrame(anomaly_score, index=delta.index, columns=['Anomaly score'])
result = pd.concat((delta,anom), axis=1)
print(result.nlargest(5,'Anomaly score'))
