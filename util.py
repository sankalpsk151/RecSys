import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from time import time


class Dataset:
    def __init__(self) -> None:
        t0 = time()
        self.ratings = pd.io.parsers.read_csv('dataset/ratings.dat', names=[
                                              'userid', 'movieid', 'ratings', 'time'], encoding='latin-1', engine='python', delimiter='::')
        self.ratings = self.ratings.drop('time', axis=1)
        print(f"Time to read from dataset is {time() - t0} seconds")

        t0 = time()
        self.split_ratings()
        # self.test_count = self.test_ratings.groupby(['userid', 'ratings'], as_index = False).size().reset_index(drop = True).reset_index(drop = True)
        # self.test_count.sort_values(by=['userid', 'ratings'], ascending = [True, False], inplace = True)
        # self.train_count = self.train_ratings.groupby(['userid', 'ratings'], as_index = False).size().reset_index(drop = True).reset_index(drop = True)
        # self.train_count.sort_values(by=['userid', 'ratings'], ascending = [True, False], inplace = True)
        # self.rtc_test = self.test_count.groupby(['userid'])['userid'].count()
        # self.rtc_train = self.train_count.groupby(['userid'])['userid'].count()
        self.test_count = self.test_ratings.groupby(['userid'])[
            'userid'].count()
        self.train_count = self.train_ratings.groupby(['userid'])[
            'userid'].count()
        print(
            f"Number of users in test and train are {len(self.test_count), len(self.train_count)}")
        print(
            f"Min movies for all users in test and train are {min(self.test_count), min(self.train_count)}")
        print(f"Time to split and sort is {time() - t0} seconds")
        # self.get_matrix()

        t0 = time()
        self.matrix = np.zeros((6040, 3952))
        for p, q, r in zip(self.train_ratings['userid'], self.train_ratings['movieid'], self.train_ratings['ratings']):
            self.matrix[p-1][q-1] = r
        print(f"Time to fill matrix {time() - t0} seconds")

        # self.movie_data = pd.io.parsers.read_csv('dataset/movies.dat', names=['movie_id', 'title', 'genre'], encoding='latin-1', engine='python', delimiter='::')
        # self.user_data = pd.io.parsers.read_csv('dataset/users.dat', names=['user_id', 'gender', 'age', 'occupation', 'zipcode'], encoding='latin-1', engine='python', delimiter='::')

    def get_matrix(self):
        matrix = pd.read_csv("dataset/matrix.csv")
        self.matrix = np.array(matrix.values)

    def split_ratings(self, test_size=0.2):
        self.train_ratings, self.test_ratings = train_test_split(
            self.ratings, test_size=test_size, stratify=self.ratings['userid'])
        # print(len(self.train_ratings['movieid'].unique()))
        # print(len(self.test_ratings['movieid'].unique()))
        self.train_ratings.sort_values(by=['userid', 'ratings'], ascending=[
                                       True, False], inplace=True)
        self.test_ratings.sort_values(by=['userid', 'ratings'], ascending=[
                                      True, False], inplace=True)
        self.train_ratings = self.train_ratings.reset_index(drop=True)
        self.test_ratings = self.test_ratings.reset_index(drop=True)
        # self.train_ratings.to_csv('dataset/train_ratings.csv')
        # self.test_ratings.to_csv('dataset/test_ratings.csv')


class EvaluttionMetrics:
    def __init__(self) -> None:
        pass

    def get_RMSE(self, y, y_pred):
        return np.sqrt(np.mean((y - y_pred)**2))

    def get_precision_on_top_k(self, y, y_pred, top_k):
        y_pred.sort_values(by=['userid', 'ratings'], ascending=[True, False])
        y.sort_values(by=['userid', 'ratings'], ascending=[True, False])
        prec_arr = np.empty(6040)  # np.array([])
        for userid in range(1, (6040+1)):
            # print(f"userid {userid}")
            userid_ratings = y[y['userid'] == userid].reset_index(drop=True).sort_values(by=['ratings'], ascending=[False])
            # print(userid_ratings['ratings'])
            top_kth_rating = userid_ratings.at[top_k-1, 'ratings']
            # print(top_kth_rating)
            val3 = userid_ratings[userid_ratings['ratings'] == top_kth_rating].reset_index(drop=True)
            s1 = set(userid_ratings['movieid'].iloc[:top_k])
            set_of_movies = s1.union(set(val3['movieid']))
            val4 = y_pred[y_pred['userid'] == userid].reset_index(drop=True).sort_values(by=['ratings'], ascending=[False])
            pred_set_of_movies = set(val4['movieid'].iloc[:top_k])
            z = len(set_of_movies.intersection(pred_set_of_movies))
            prec_arr[userid - 1] = z/top_k

        return np.mean(prec_arr)
        # map_k = 0
        # for movie_count in count:
        #     _temp = np.argsort(y_pred[x:x+movie_count])[::-1]
        #     for i in range(1, top_k + 1):
        #         map_k += (_temp[:i] < i).sum() / i
        #     x += movie_count
        # return map_k / (len(count) * top_k)

    def get_spearman_rank(y, y_pred):
        n = len(y)
        rank_x = {value: i for i, value in enumerate(sorted((y)), 1)}
        rank_y = {value: i for i, value in enumerate(sorted((y_pred)), 1)}
        rank_x = [rank_x[value] for value in y]
        rank_y = [rank_y[value] for value in y_pred]
        d = [(rank_x[i] - rank_y[i])**2 for i in range(n)]
        return 1 - (6 * sum(d)) / (n * (n**2 - 1))
