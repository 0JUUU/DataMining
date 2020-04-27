import matplotlib.pyplot as plt 
import pandas as pd 

data = pd.read_csv('8.chameleon.data.csv',  delimiter=' ', names = ['x', 'y'])
data.plot.scatter(x = 'x', y = 'y')
plt.show()