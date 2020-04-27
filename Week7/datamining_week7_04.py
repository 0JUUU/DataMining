import pandas as pd 

ratings = [['john',5,5,2,1],['mary',4,5,3,2],['bob',4,4,4,3],['lisa',2,2,4,5],['lee',1,2,3,4],['harry',2,1,5,5]]
titles = ['user', 'Jaws', 'Star Wars', 'Exorcist', 'Omen']
movies = pd.DataFrame(ratings,columns=titles)

from sklearn import cluster
data = movies.drop('user', axis = 1)
k_means = cluster.KMeans(n_clusters = 2, max_iter = 50, random_state = 1)
k_means.fit(data)
labels = k_means.labels_
pd.DataFrame(labels, index = movies.user, columns = ['Cluster ID'])

centroids = k_means.cluster_centers_

import numpy as np

testData = np.array([[4,5,1,2],[3,2,4,4],[2,3,4,1],[3,2,3,3],[5,4,1,4]])
labels = k_means.predict(testData)
labels = labels.reshape(-1,1)
usernames = np.array(['paul','kim','liz','tom','bill']).reshape(-1,1)
cols = movies.columns.tolist()
cols.append('Cluster ID')
newusers = pd.DataFrame(np.concatenate((usernames, testData, labels), axis = 1), columns = cols)

print(newusers)