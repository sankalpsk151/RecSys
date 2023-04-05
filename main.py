import numpy as np
import pandas as pd
# from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self) -> None:
        self.ratings = pd.io.parsers.read_csv('dataset/ratings.dat', names=[
                                              'userid', 'movieid', 'ratings', 'time'], encoding='latin-1', engine='python', delimiter='::')
        self.ratings = self.ratings.drop('time', axis=1)
        self.split_ratings()
        count = self.test_ratings.groupby(['userid'])['userid'].count()
        print(min(count))
        # self.matrix = np.zeros((6040, 3952))
        # for i, j in zip(self.train_ratings['userid'], self.train_ratings['movieid']):
        #     self.matrix[i-1][j-1] = self.train_ratings['ratings']
        # self.movie_data = pd.io.parsers.read_csv('dataset/movies.dat', names=['movie_id', 'title', 'genre'], encoding='latin-1', engine='python', delimiter='::')
        # self.user_data = pd.io.parsers.read_csv('dataset/users.dat', names=['user_id', 'gender', 'age', 'occupation', 'zipcode'], encoding='latin-1', engine='python', delimiter='::')

    def get_matrix(self):
        matrix = pd.read_csv("dataset/matrix.csv")
        self.matrix = np.array(matrix.values)

    def split_ratings(self, test_size=0.2):
        self.train_ratings, self.test_ratings = train_test_split(
            self.ratings, test_size=test_size, stratify=self.ratings['userid'])
        self.test_ratings.sort_values(by=['userid'], inplace=True)
        self.train_ratings = self.train_ratings.reset_index(drop=True)
        self.test_ratings = self.test_ratings.reset_index(drop=True)
        self.train_ratings.to_csv('dataset/train_ratings.csv')
        self.test_ratings.to_csv('dataset/test_ratings.csv')


class ColabrativeFiltering:
    def __init__(self, matrix, k=10):
        self.matrix = matrix
        self.k = k

    def get_correlation_coef(x, y):
        return (1 - np.corrcoef(x, y)[0, 1])

    def get_top_k_users(self, userid):
        dist = []
        for i in range(self.matrix.shape[0]):
            dist.append(self.get_correlation_coef(
                self.matrix[userid - 1], self.matrix[i]))
        dist = np.array(dist)
        sorted = np.argsort(dist)[:self.k]
        return sorted, [dist[i] for i in sorted]

    def get_rating(self, userid, movieid):
        top_k_users, dist = self.get_top_k_users(userid)
        rating = 0
        for i in range(len(self.k)):
            rating += self.matrix[top_k_users[i] -
                                  1][movieid - 1] * (1 - dist[i])
        return rating / (len(self.k)-sum(dist))

    def get_RMSE(self, test_ratings):
        rmse = 0
        for i, j in zip(test_ratings['userid'], test_ratings['movieid']):
            rmse += (self.get_rating(i, j) - test_ratings['ratings'])**2
        return np.sqrt(rmse / len(test_ratings))

    def get_precision_on_top_k(self, test_ratings, k):
        precision = 0
        count = test_ratings.group_by(['userid'])['userid'].count()[1]
        print(count)
        for i in range(1, len(count)+1):
            for j in test_ratings['movieid']:

        for i, j in zip(test_ratings['userid'], test_ratings['movieid']):
            top_k_users, dist = self.get_top_k_users(i)
            if j in top_k_users:
                precision += 1
        return precision / len(test_ratings)

    def get_spearman_rank(self, test_ratings):
        rank = []
        for i, j in zip(test_ratings['userid'], test_ratings['movieid']):
            top_k_users, dist = self.get_top_k_users(i)
            rank.append(top_k_users.index(j))
            return np.mean(rank)


data = Dataset()
