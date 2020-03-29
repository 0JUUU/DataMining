import numpy as np 

X = np.random.randn(5,3)
print(X)

C = X.T.dot(X)

invC = np.linalg.inv(C)
print(invC)
detC = np.linalg.det(C)
print(detC)
S, U = np.linalg.eig(C)
print(S)
print(U)