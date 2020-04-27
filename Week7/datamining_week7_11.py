import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 

data = pd.read_csv('8.chameleon.data.csv',  delimiter=' ', names = ['x', 'y'])

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=15.5, min_samples=5).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = pd.DataFrame(db.labels_,columns=['Cluster ID'])
result = pd.concat((data,labels), axis=1)
result.plot.scatter(x='x',y='y',c='Cluster ID', colormap='jet')


plt.show()