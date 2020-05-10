import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy.linalg import inv

stocks = pd.read_csv('9.stocks.csv', header='infer' ) 
stocks.index = stocks['Date']
stocks = stocks.drop(['Date'],axis=1)
stocks.head()

N,d = stocks.shape
delta = pd.DataFrame(100*np.divide(stocks.iloc[1:,:].values-stocks.iloc[:N-1,:].values,
        stocks.iloc[:N-1,:].values), columns=stocks.columns, index=stocks.iloc[1:].index)

knn = 4
nbrs = NearestNeighbors(n_neighbors=knn, metric=distance.euclidean).fit(delta.as_matrix())
distances, indices = nbrs.kneighbors(delta.as_matrix())

anomaly_score = distances[:,knn-1]

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(delta.MSFT,delta.F,delta.BAC,c=anomaly_score,cmap='jet')
ax.set_xlabel('Microsoft')
ax.set_ylabel('Ford')
ax.set_zlabel('Bank of America')
fig.colorbar(p)
plt.show()