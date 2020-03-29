import numpy as np 

X = np.random.randn(2,3)
print(X)
print(X.T)

y = np.random.randn(3)
print(y)
print(X.dot(y))
print(X.dot(X.T))
print(X.T.dot(X))