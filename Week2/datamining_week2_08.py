import numpy as np 

my2darr = np.arange(1,13,1).reshape(3,4)
print(my2darr)

divBy3 = my2darr[my2darr % 3 == 0]
print(divBy3, type(divBy3))

divBy3LastRow = my2darr[2:, my2darr[2,:] % 3 == 0]
print(divBy3LastRow)