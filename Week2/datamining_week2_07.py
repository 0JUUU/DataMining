import numpy as np 

my2dlist = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
print(my2dlist)
print(my2dlist[2])
print(my2dlist[:][2])
# print(my2dlist[:,2])      # syntax error

my2darr = np.array(my2dlist)
print(my2darr)
print(my2darr[2][:])
print(my2darr[2,:])
print(my2darr[:][2])
print(my2darr[:,2])
print(my2darr[:2,2:])