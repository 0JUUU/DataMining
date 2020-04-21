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

testData = [['gila monster',0,0,0,0,1,1,'non-mammals'],
            ['platypus',1,0,0,0,1,1,'mammals'],
            ['owl',1,0,0,1,1,0,'non-mammals'],
            ['dolphin',1,1,1,0,0,0,'mammals']]
testData = pd.DataFrame(testData, columns=data.columns)

testY = testData['Class']
testX = testData.drop(['Name', 'Class'], axis=1)

predY = clf.predict(testX)
predictions = pd.concat([testData['Name'], pd.Series(predY, name = 'Predicted Class')], axis = 1)
print(predictions)