import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from time import time
from util import Dataset, EvaluttionMetrics


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
        self.matrix = np.subtract(
            self.matrix, self.global_mean, where=self.bool_mat)
        # print(self.matrix)
        self.matrix = np.subtract(
            self.matrix, self.movie_deviation, where=self.bool_mat)
        # print(self.matrix)
        self.get_top_k_users()
        self.get_results()

    def find_global_mean(self):
        self.global_mean = np.mean(self.matrix, where=self.bool_mat)
        print(self.global_mean)

    def find_user_deviation(self):
        self.user_deviation = np.mean(
            self.matrix, where=self.bool_mat, axis=1) - self.global_mean

    def find_movie_deviation(self):
        self.movie_deviation = np.nanmean(
            self.matrix, where=self.bool_mat, axis=0) - self.global_mean
        np.nan_to_num(self.movie_deviation, copy=False)

    def get_rating(self, userid, movieid):
        rating = np.average(self.matrix[self.top_k_users[userid - 1], movieid - 1].reshape(
            self.k,), weights=self.top_k_sim[userid - 1])
        return rating + self.global_mean + self.movie_deviation[movieid - 1]


if __name__ == "__main__":
    data = Dataset()
    ev = EvaluttionMetrics()

    print("======== Collabrative filtering =========")
    cf = ColabrativeFiltering(
        data.matrix, data.train_ratings, data.test_ratings, 15)
    print(
        f"Training RMSE for Collabrative filtering: {ev.get_RMSE(cf.train_ratings['ratings'], cf.pred_train)}")
    print(
        f"Test RMSE for Collabrative filtering: {ev.get_RMSE(cf.test_ratings['ratings'], cf.pred_test)}")

    pred_test_df = pd.DataFrame()
    pred_test_df['movieid'] = cf.test_ratings['movieid']
    pred_test_df['userid'] = cf.test_ratings['userid']
    pred_test_df['ratings'] = cf.pred_test

    pred_train_df = pd.DataFrame()
    pred_train_df['movieid'] = cf.train_ratings['movieid']
    pred_train_df['userid'] = cf.train_ratings['userid']
    pred_train_df['ratings'] = cf.pred_train

    print(
        f"Training precision on topk: {ev.get_precision_on_top_k(cf.train_ratings, pred_train_df, 4)}")
    print(
        f"Test precision on topk: {ev.get_precision_on_top_k(cf.test_ratings, pred_test_df, 4)}")

    print("================= Collabrative filtering with Baseline ==================")
    cfb = CollaborativeWithBaseline(
        data.matrix, data.train_ratings, data.test_ratings, 15)
    print(
        f"Training RMSE for Collabrative filtering with Baseline: {ev.get_RMSE(cfb.train_ratings['ratings'], cfb.pred_train)}")
    print(
        f"Test RMSE for Collabrative filtering with Baseline: {ev.get_RMSE(cfb.test_ratings['ratings'], cfb.pred_test)}")


# top_k = 10
# print(
#     f"Training MAP@{top_k} for Collabrative filtering: {ev.get_precision_on_top_k(cf.train_ratings['ratings'], cf.pred_train, data.train_count, top_k)}")
# top_k = 4
# print(
#     f"Test MAP@{top_k} for Collabrative filtering: {ev.get_precision_on_top_k(cf.test_ratings['ratings'], cf.pred_test, data.test_count, top_k)}")
