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

plt.scatter(y_test, y_pred_test, color='black')
plt.title('Comparing true and predicted values for test set')
plt.xlabel('True values for y')
plt.ylabel('Predicted values for y')

plt.show()
print("Root mean squared error = %.4f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))
print('R-squared = %.4f' % r2_score(y_test, y_pred_test))
