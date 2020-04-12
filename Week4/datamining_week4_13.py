import numpy as np 
import pandas as pd

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases//breast-cancer-wisconsin/breast-cancer-wisconsin.data',
header=None)
data.columns = ['Sample code', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of CellShape', 
'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses','Class']
data = data.drop(['Sample code'],axis=1) 
print(data.head())