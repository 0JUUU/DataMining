import numpy as np 
from pandas import DataFrame

npdata = np.random.randn(5,3)
columnNames =['x1','x2','x3']
data = DataFrame(npdata, columns = columnNames)

print('data =')
print(data)

columnNames = ['x1','x2','x3']
data2 = DataFrame(np.random.randn(5,3), columns = columnNames)
print('\ndata2 =')
print(data2)

print('\ndata + data2 = ')
print(data.add(data2))

print('\ndata * data2 = ')
print(data.mul(data2))