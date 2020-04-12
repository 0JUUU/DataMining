import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
import pandas as pd

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases//breast-cancer-wisconsin/breast-cancer-wisconsin.data',
header=None)
data.columns = ['Sample code', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of CellShape', 
'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses','Class']
data = data.replace('?', np.NaN)

data2 = data.drop(['Class'],axis=1)
data2['Bare Nuclei'] = pd.to_numeric(data2['Bare Nuclei'])
Z = (data2 - data2.mean())/data2.std()
print(Z[20:25])