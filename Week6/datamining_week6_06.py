import pandas as pd
from sklearn import tree
import pydotplus
from IPython.display import Image

data = pd.read_csv('6.vertebrate.csv',header='infer')
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')

testData = [['gila monster',0,0,0,0,1,1,'non-mammals'],
            ['platypus',1,0,0,0,1,1,'mammals'],
            ['owl',1,0,0,1,1,0,'non-mammals'],
            ['dolphin',1,1,1,0,0,0,'mammals']]
testData = pd.DataFrame(testData, columns=data.columns)
print(testData)
