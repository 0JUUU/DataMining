import pandas as pd
from sklearn import tree
import pydotplus
from IPython.display import Image

data = pd.read_csv('6.vertebrate.csv',header='infer')
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')

Y = data['Class']
X = data.drop(['Name','Class'], axis = 1)

clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
clf = clf.fit(X, Y)

dot_data = tree.export_graphviz(clf, feature_names = X.columns, 
            class_names = ['mammals','non-mammals'], filled = True, out_file = None)

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())
graph.write_png('5.png')