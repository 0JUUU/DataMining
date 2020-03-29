import matplotlib.pyplot as plt
import matplotlib 
from pandas import DataFrame

tuplelist = [(2011,45.1,32.4),(2012,42.4,34.5),(2013,47.2,39.2),
 (2014,44.2,31.4),(2015,39.9,29.8),(2016,41.5,36.7)]
columnNames = ['year','temp','precip']
weatherData = DataFrame(tuplelist, columns=columnNames)
weatherData[['temp','precip']].plot(kind='box', title='Box plot')

plt.show() 