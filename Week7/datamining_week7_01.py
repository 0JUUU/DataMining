import pandas as pd 

ratings = [['john',5,5,2,1],['mary',4,5,3,2],['bob',4,4,4,3],['lisa',2,2,4,5],['lee',1,2,3,4],['harry',2,1,5,5]]
titles = ['user', 'Jaws', 'Star Wars', 'Exorcist', 'Omen']
movies = pd.DataFrame(ratings,columns=titles)


print(movies)