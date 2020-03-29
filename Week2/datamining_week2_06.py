import numpy as np 

x = np.arange(-5,5)
print(x)

y = x[3:5]
print(y)
y[:] = 1000
print(y)
print(x)

z = x[3:5].copy()
print(z)
z[:] = 500
print(z)
print(x)