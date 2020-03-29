import numpy as np 
from pandas import Series

s2 = Series(np.random.randn(6))
print(s2)
print('Values=',s2.values)
print('Index=',s2.index)