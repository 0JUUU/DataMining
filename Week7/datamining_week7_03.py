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
print(pd.DataFrame(centroids, columns = data.columns))