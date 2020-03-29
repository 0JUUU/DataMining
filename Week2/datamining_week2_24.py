from pandas import DataFrame

cars = {'make': ['Ford','Honda','Toyota','Tesla'],
        'model': ['Taurus','Accord','Camry','Model S'],
        'MSRP': [27595, 23570, 23495, 68000]}
carData = DataFrame(cars)
print(carData)