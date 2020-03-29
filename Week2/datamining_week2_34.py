import numpy as np 
from pandas import DataFrame

npdata = np.random.randn(5,3)
columnNames =['x1','x2','x3']
data = DataFrame(npdata, columns = columnNames)
print(data)

print('Data transpose operation:')
print(data.T)

print('Addition:')
print(data + 4)

print('Multiplication:')
print(data*10)