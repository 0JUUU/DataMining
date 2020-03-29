import numpy as np

my2darr = np.arange(1,13,1).reshape(4,3)
print(my2darr)

indices = [2,1,0,3]
print(my2darr[indices,:])

rowIndex = [0,0,1,2,3]
columnIndex = [0,2,0,1,2]
print(my2darr[rowIndex, columnIndex])