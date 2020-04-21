import pandas as pd

data = pd.read_csv('6.vertebrate.csv',header='infer')

data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')
print(data)