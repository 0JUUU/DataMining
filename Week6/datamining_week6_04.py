import pandas as pd
from sklearn import tree

data = pd.read_csv('6.vertebrate.csv',header='infer')
Y = data['Class']
X = data.drop(['Name','Class'], axis = 1)
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')

clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
clf = clf.fit(X, Y)