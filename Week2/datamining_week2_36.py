import numpy as np 
from pandas import DataFrame

npdata = np.random.randn(5,3)
columnNames =['x1','x2','x3']
data = DataFrame(npdata, columns = columnNames)

print(data.abs())

print('\nMaximum value per column:')
print(data.max()) 
print('\nMinimum value per row:')
print(data.min(axis=1)) 
print('\nSum of values per column:')
print(data.sum()) 
print('\nAverage value per row:')
print(data.mean(axis=1)) 
print('\nCalculate max - min per column')
f = lambda x: x.max() - x.min()
print(data.apply(f))
print('\nCalculate max - min per row')
f = lambda x: x.max() - x.min()
print(data.apply(f, axis=1))