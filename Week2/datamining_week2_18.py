from pandas import Series

capitals = {'M1':'Lansing','CA':'Sacramento','TX':'Austin','MN':'St Paul'}

s4 = Series(capitals)
print(s4)
print('Values=',s4.values)
print('Index=',s4.index)