import matplotlib.pyplot as plt
import matplotlib 
from pandas import Series

s3 = Series([1.2,2.5,-2.2,3.1,-0.8,-3.2,1.4],
 index = ['Jan 1','Jan 2','Jan 3','Jan 4','Jan 5','Jan 6','Jan 7'])
s3.plot(kind = 'hist', title = 'Histogram')
plt.show() 