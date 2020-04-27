import pandas as pd 

data = pd.read_csv('8.vertebrate.csv', header = 'infer')

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt 

names = data['Name']
Y = data['Class']
X = data.drop(['Name', 'Class'], axis = 1)

Z = hierarchy.linkage(X.as_matrix(), 'complete')
dn = hierarchy.dendrogram(Z, labels = names.tolist(), orientation='right')
plt.show()