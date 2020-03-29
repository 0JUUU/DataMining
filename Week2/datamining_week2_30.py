import numpy as np 
from pandas import DataFrame

npdata = np.random.randn(5,3)
columnNames =['x1','x2','x3']
data = DataFrame(npdata, columns = columnNames)
cars = {'make': ['Ford','Honda','Toyota','Tesla'],
        'model': ['Taurus','Accord','Camry','Model S'],
        'MSRP': [27595, 23570, 23495, 68000]}
carData2 = DataFrame(cars, index = [1,2,3,4])
print('Row 3 of data table:')
print(data.iloc[2])
print(type(data.iloc[2]))
print('\nRow 3 of car data table:')
print(carData2.iloc[2])