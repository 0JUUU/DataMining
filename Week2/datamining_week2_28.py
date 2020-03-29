import numpy as np 
from pandas import DataFrame

npdata = np.random.randn(5,3)
columnNames =['x1','x2','x3']
data = DataFrame(npdata, columns = columnNames)
print(data)