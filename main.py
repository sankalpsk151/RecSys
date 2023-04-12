import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
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
        print(len(self.train_ratings['movieid'].unique()))
        print(len(self.test_ratings['movieid'].unique()))
        self.train_ratings.sort_values(by=['userid', 'ratings'], ascending = [True, False], inplace=True)
        self.test_ratings.sort_values(by=['userid', 'ratings'], ascending = [True, False],inplace=True)
        self.train_ratings = self.train_ratings.reset_index(drop=True)
        self.test_ratings = self.test_ratings.reset_index(drop=True)
        # self.train_ratings.to_csv('dataset/train_ratings.csv')
        # self.test_ratings.to_csv('dataset/test_ratings.csv')


class ColabrativeFiltering:
    def __init__(self, matrix, train, test, k=10):
        self.matrix = matrix
        self.k = k
        self.train_ratings = train
        self.test_ratings = test
        self.get_top_k_users()
        self.get_results()

    def get_top_k_users(self):
        row_sums = np.linalg.norm(self.matrix, axis=1)
        sim_mat = np.matmul(self.matrix, self.matrix.T) / \
            np.matmul(row_sums[:, np.newaxis], row_sums[:, np.newaxis].T)
        self.top_k_users = np.argsort(sim_mat, axis=1)[:, -(self.k + 1):-1]
        self.top_k_sim = np.take_along_axis(sim_mat, self.top_k_users, axis=1)

    def get_rating(self, userid, movieid):
        rating = np.average(self.matrix[self.top_k_users[userid - 1], movieid - 1].reshape(
            self.k,), weights=self.top_k_sim[userid - 1])
        return rating

    def get_results(self):
        t0 = time()
        self.pred_train = [self.get_rating(i, j) for i, j in zip(
            self.train_ratings['userid'], self.train_ratings['movieid'])]
        print(
            f"Time taken to predict using Collabrative Filtering Train: {time() - t0} seconds")

        t0 = time()
        self.pred_test = [self.get_rating(i, j) for i, j in zip(
            self.test_ratings['userid'], self.test_ratings['movieid'])]
        print(
            f"Time taken to predict using Collabrative Filtering Test: {time() - t0} seconds")

class CollaborativeWithBaseline(ColabrativeFiltering):
    def __init__(self, matrix, train, test, k=10):
        self.matrix = np.array(matrix)
        self.k = k
        self.train_ratings = train
        self.test_ratings = test
        self.bool_mat = np.where(self.matrix == 0, False, True)
        self.find_global_mean()
        self.find_user_deviation()
        self.find_movie_deviation()
        self.matrix = np.subtract(self.matrix, self.global_mean, where=self.bool_mat)
        # print(self.matrix)
        self.matrix = np.subtract(self.matrix, self.movie_deviation, where=self.bool_mat)
        # print(self.matrix)
        self.get_top_k_users()
        self.get_results()

    def find_global_mean(self):
        self.global_mean = np.mean(self.matrix, where=self.bool_mat)
        print(self.global_mean)

    def find_user_deviation(self):
        self.user_deviation = np.mean(self.matrix, where= self.bool_mat, axis=1) - self.global_mean

    def find_movie_deviation(self):
        self.movie_deviation = np.nanmean(self.matrix, where=self.bool_mat, axis=0) - self.global_mean
        np.nan_to_num(self.movie_deviation, copy=False)
    
    def get_rating(self, userid, movieid):
        rating = np.average(self.matrix[self.top_k_users[userid - 1], movieid - 1].reshape(
            self.k,), weights=self.top_k_sim[userid - 1])
        return rating + self.global_mean + self.movie_deviation[movieid - 1]
    
    

class EvaluttionMetrics:
    def __init__(self) -> None:
        pass

    def get_RMSE(self, y, y_pred):
        return np.sqrt(np.mean((y - y_pred)**2))

    def get_precision_on_top_k(self, y, y_pred, count, top_k):
        x = 0
        map_k = 0
        for movie_count in count:
            _temp = np.argsort(y_pred[x:x+movie_count])[:-1]
            for i in range(1, top_k + 1):
                map_k += (_temp[:i] < i).sum() / i
            x += movie_count
        return map_k / (len(count) * top_k)

    def get_spearman_rank(self, test_ratings):
        rank = []
        for i, j in zip(test_ratings['userid'], test_ratings['movieid']):
            top_k_users, dist = self.get_top_k_users(i)
            rank.append(top_k_users.index(j))
            return np.mean(rank)


# data = Dataset()
# cf = ColabrativeFiltering(
#     data.matrix, data.train_ratings, data.test_ratings, 15)
# ev = EvaluttionMetrics()
# print(
#     f"Training RMSE for Collabrative filtering: {ev.get_RMSE(cf.train_ratings['ratings'], cf.pred_train)}")
# print(
#     f"Test RMSE for Collabrative filtering: {ev.get_RMSE(cf.test_ratings['ratings'], cf.pred_test)}")
# top_k = 10
# print(
#     f"Training MAP@{top_k} for Collabrative filtering: {ev.get_precision_on_top_k(cf.train_ratings['ratings'], cf.pred_train, data.train_count, top_k)}")
# top_k = 4
# print(
#     f"Test MAP@{top_k} for Collabrative filtering: {ev.get_precision_on_top_k(cf.test_ratings['ratings'], cf.pred_test, data.test_count, top_k)}")
if __name__ == "__main__":
    data = Dataset()
    ev = EvaluttionMetrics()
    
    print("======== Collabrative filtering =========")
    cf = ColabrativeFiltering(data.matrix, data.train_ratings, data.test_ratings, 15)
    print(
    f"Training RMSE for Collabrative filtering: {ev.get_RMSE(cf.train_ratings['ratings'], cf.pred_train)}")
    print(
    f"Test RMSE for Collabrative filtering: {ev.get_RMSE(cf.test_ratings['ratings'], cf.pred_test)}")
    
    print("================= Collabrative filtering with Baseline ==================")
    cfb = CollaborativeWithBaseline(data.matrix, data.train_ratings, data.test_ratings, 15)
    print(
        f"Training RMSE for Collabrative filtering with Baseline: {ev.get_RMSE(cfb.train_ratings['ratings'], cfb.pred_train)}")
    print(
        f"Test RMSE for Collabrative filtering with Baseline: {ev.get_RMSE(cfb.test_ratings['ratings'], cfb.pred_test)}")