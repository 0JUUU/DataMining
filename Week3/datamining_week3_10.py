import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.plotting import parallel_coordinates

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
data.columns = ['sepal length', 'sepa width', 'petal length', 'patal width', 'class']

parallel_coordinates(data,'class')
plt.show()