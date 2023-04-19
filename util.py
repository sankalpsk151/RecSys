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
        print("y_pred:\n", y_pred)
        print(y)
        prec_arr = np.empty(6040)  # np.array([])
        for userid in range(1, (6040+1)):
            # print(f"userid {userid}")
            userid_ratings = y[y['userid'] == userid].reset_index(
                drop=True).sort_values(by=['ratings'], ascending=[False])
            # print(userid_ratings['ratings'])
            top_kth_rating = userid_ratings.at[top_k-1, 'ratings']
            # print(top_kth_rating)
            val3 = userid_ratings[userid_ratings['ratings']
                                  == top_kth_rating].reset_index(drop=True)
            s1 = set(userid_ratings['movieid'].iloc[:top_k])
            set_of_movies = s1.union(set(val3['movieid']))
            val4 = y_pred[y_pred['userid'] == userid].reset_index(
                drop=True).sort_values(by=['ratings'], ascending=[False])
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

    def precision_top_k(self, y, y_pred, k):
        """Calculates precision at top k for predictions

        Tkaes average of precision for all users.
        """
        def is_relevant(rating):
            return rating > 3
        print("Calculating precision\n")
        y_pred.sort_values(by=['userid', 'ratings'], ascending=[True, False])
        y.sort_values(by=['userid', 'ratings'], ascending=[True, False])
        # print(y_pred)
        # print(y)
        precisions_total = 0
        for userid in range(1, (6040+1)):
            ratings = y[y['userid'] == userid].reset_index(
                drop=True).sort_values(by=['ratings'], ascending=[False])
            pred_ratings = y_pred[y_pred['userid'] == userid].reset_index(
                drop=True).sort_values(by=['ratings'], ascending=[False])
            local_correct = 0
            for i in range(k):
                if is_relevant(pred_ratings.iloc[i]['ratings']) and is_relevant(ratings.iloc[i]['ratings']):
                    local_correct += 1
            precisions_total += local_correct / k
        return precisions_total / 6040

    def spearman_coef(self, y, y_pred):
        """Finds the average spearman coefficient over all users

        Parameters:
        y (numpy vector): Ratings
        y_pred (numpy vector): Predicted Ratings

        Returns:
        spearman_coefficient (float) : Between -1 and 1        
        """
        total_coef = 0
        for userid in range(1, (6040+1)):
            ratings = y[y['userid'] == userid].reset_index(
                drop=True).sort_values(by=['ratings'], ascending=[False])
            pred_ratings = y_pred[y_pred['userid'] == userid].reset_index(
                drop=True).sort_values(by=['ratings'], ascending=[False])
            total_coef += spearman_with_ties(
                np.array(ratings['ratings']), np.array(pred_ratings['ratings']))
        return total_coef / 6040


def spearman_with_ties(x, y):
    """
    Calculates Spearman's rank correlation coefficient with ties for two arrays x and y.

    """
    n = len(x)

    ranks_x = np.argsort(np.argsort(x)) + 1  # Assign ranks to x
    ranks_y = np.argsort(np.argsort(y)) + 1  # Assign ranks to y
    ranks_x = ranks_x.astype(float)
    ranks_y = ranks_y.astype(float)

    # Calculate the average ranks for tied values
    unique_x, counts_x = np.unique(x, return_counts=True)

    for i in range(len(unique_x)):
        val = unique_x[i]
        if counts_x[i] > 1:
            tied_ranks = ranks_x[x == val]
            avg_rank = np.mean(tied_ranks)
            ranks_x[x == val] = avg_rank

    unique_y, counts_y = np.unique(y, return_counts=True)
    for i in range(len(unique_y)):
        val = unique_y[i]
        if counts_y[i] > 1:
            tied_ranks = ranks_y[y == val]
            # print(tied_ranks)
            avg_rank = np.mean(tied_ranks)
            # print(avg_rank)
            ranks_y[y == val] = avg_rank
            # print(ranks_y[y == val])

    # print(ranks_x.shape)

    # Calculate the differences between ranks
    rank_diffs = ranks_x - ranks_y
    rank_diffs_squared = rank_diffs ** 2

    # Calculate Spearman's rank correlation coefficient
    sum_rank_diffs_squared = np.sum(rank_diffs_squared, where=(x != 0))
    spearman_coefficient = 1 - (6 * sum_rank_diffs_squared) / (n * (n**2 - 1))

    return spearman_coefficient
