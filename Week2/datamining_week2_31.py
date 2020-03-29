from pandas import DataFrame

cars = {'make': ['Ford','Honda','Toyota','Tesla'],
        'model': ['Taurus','Accord','Camry','Model S'],
        'MSRP': [27595, 23570, 23495, 68000]}
carData2 = DataFrame(cars, index = [1,2,3,4])
print(carData2.iloc[1,2])
print(carData2.loc[1,'model'])

print('carData2.iloc[1:3][1:3]=')
print(carData2.iloc[1:3,1:3])