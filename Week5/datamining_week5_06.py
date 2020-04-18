import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

seed = 1
numInstances = 200
np.random.seed(seed)
X = np.random.rand(numInstances,1).reshape(-1,1)
y_true = -3 * X + 1
y = y_true + np.random.normal(size = numInstances).reshape(-1,1)

numTrain = 20
numTest = numInstances - numTrain

X_train = X[:-numTest]
X_test = X[-numTest:]
y_train = y[:-numTest]
y_test = y[-numTest:]

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

y_pred_test = regr.predict(X_test)

print('Slope = ', regr.coef_[0][0])
print('Intercept = ', regr.intercept_[0])

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred_test, color='blue', linewidth=3)
titlestr = 'Predicted Function: y = %.2fX + %.2f' % (regr.coef_[0], regr.intercept_[0])
plt.title(titlestr)
plt.xlabel('X')
plt.ylabel('y')
plt.show()
