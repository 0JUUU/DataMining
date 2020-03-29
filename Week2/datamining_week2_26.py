from pandas import DataFrame

cars = {'make': ['Ford','Honda','Toyota','Tesla'],
        'model': ['Taurus','Accord','Camry','Model S'],
        'MSRP': [27595, 23570, 23495, 68000]}
carData2 = DataFrame(cars, index = [1,2,3,4])
carData2['year'] = 2018
carData2['dealership'] = ['Courtesy Ford','Captial Honda','Spartan Toyota','N/A']
print(carData2)