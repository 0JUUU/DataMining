import pandas as pd
from pandas.api.types import is_numeric_dtype

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
data.columns = ['sepal length', 'sepa width', 'petal length', 'patal width', 'class']

for col in data.columns:
    if is_numeric_dtype(data[col]):
        print('%s: ' % (col))
        print('\t Mean = %.2f' % data[col].mean())
        print('\t Standard deviation = %.2f' % data[col].std())
        print('\t Minimum = %.2f' % data[col].min())
        print('\t Maximum = %.2f' % data[col].max())