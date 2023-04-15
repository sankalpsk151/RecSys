import pandas as pd

ratings = pd.io.parsers.read_csv('dataset/ratings.dat', names=[
                                              'userid', 'movieid', 'ratings', 'time'], encoding='latin-1', engine='python', delimiter='::')
ratings = ratings.drop('time', axis=1)
ratings.sort_values(by=['userid', 'ratings'])
userid = 2
userid_ratings = ratings[ratings['userid'] == userid].reset_index(drop=True).sort_values(by=['ratings'])
print(userid_ratings.head(129))
top_k  = 5
top_kth_rating = userid_ratings.at[top_k-1, 'ratings']
val3 = userid_ratings[userid_ratings['ratings'] == top_kth_rating].reset_index(drop=True)
print(val3)
s1 = set(userid_ratings['movieid'].iloc[:top_k])
set_of_movies = s1.union(set(val3['movieid']))
print(set_of_movies)