import pandas as pd

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
data.columns = ['sepal length', 'sepa width', 'petal length', 'patal width', 'class']

data.head()

print(data.head())