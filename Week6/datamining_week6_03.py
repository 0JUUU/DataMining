import pandas as pd

data = pd.read_csv('6.vertebrate.csv',header='infer')

data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')
cross_tab = pd.crosstab([data['Warm-blooded'], data['Gives Birth']], data['Class'])

print(cross_tab)