import pandas as pd
import matplotlib.pyplot as plt 
data1 = pd.read_csv('8.2d_data.csv', delimiter=' ', names=['x','y'])
data2 = pd.read_csv('8.elliptical.csv', delimiter=' ', names=['x','y'])

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
data1.plot.scatter(x='x',y='y',ax=ax1)
data2.plot.scatter(x='x',y='y',ax=ax2)

plt.show()