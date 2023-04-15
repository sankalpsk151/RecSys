from timeit import default_timer as timer
import pandas as pd
import numpy as np


class Dataset:
    def __init__(self):
        self.original_df = pd.io.parsers.read_csv('dataset/ratings.dat', names=[
                                                  'userid', 'movieid', 'ratings', 'time'], encoding='latin-1', engine='python', delimiter='::')
        self.get_train_test_df()
        self.get_train_matrix()

    def get_train_test_df(self):
        self.train_df = pd.read_csv("dataset/train_ratings.csv")
        self.test_df = pd.read_csv("dataset/test_ratings.csv")
        self.train_df = self.train_df.drop('Unnamed: 0', axis=1)
        self.test_df = self.test_df.drop('Unnamed: 0', axis=1)
        return self.train_df, self.test_df

    def get_train_matrix(self):
        self.matrix = pd.read_csv("dataset/matrix.csv")
        self.matrix = self.matrix.drop('Unnamed: 0', axis=1)
        self.matrix.columns = self.matrix.columns.astype(int)
        self.movies_map = self.matrix.columns
        self.matrix_np = self.matrix.to_numpy()

        self.movies_d = {}
        for i in range(len(self.movies_map)):
            self.movies_d[self.movies_map[i]] = i
        return self.matrix_np, self.movies_d

    def RMSE_training(self, predictionMatrix):
        return np.sqrt(((self.matrix_np - predictionMatrix)**2).sum(where=self.matrix_np != 0) / np.count_nonzero(self.matrix_np))

    def RMSE_testing(self, predictionMatrix):
        # Iterate through all rows of ds.test_df
        # Get the movie id and the rating
        total_error = 0
        for index, row in self.test_df.iterrows():
            # Get the movie id and the rating
            movie_id = row['movieid']
            rating = row['ratings']
            user_index = row['userid'] - 1
            # Get the index of the movie id
            movie_index = self.movies_d[movie_id]
            # Get the predicted rating
            predicted_rating = predictionMatrix[user_index][movie_index]
            # Calculate the error
            error = (rating - predicted_rating)**2
            # Add the error to the total error
            total_error += error
        return np.sqrt(total_error / len(self.test_df))

    def print_metrics(self, predictionMatrix):
        print(f"Training RMSE: {self.RMSE_training(predictionMatrix)}")
        print(f"Testing RMSE: {self.RMSE_testing(predictionMatrix)}")


class CUR:
    def __init__(self, matrix):
        self.A = matrix
        self.matrix = matrix
        self.C = None
        self.U = None
        self.R = None
        self.cur_approx = None

    def get_probabilities(self):
        A = self.A
        row_prob = np.sum(A**2, axis=1)
        row_prob = row_prob/np.sum(row_prob)
        col_prob = np.sum(A**2, axis=0)
        col_prob = col_prob/np.sum(col_prob)
        return row_prob, col_prob

    def get_C_R_W(self, r):
        A = self.A
        row_prob, col_prob = self.get_probabilities()
        row_indices = np.random.choice(
            A.shape[0], r, replace=False, p=row_prob)
        col_indices = np.random.choice(
            A.shape[1], r, replace=False, p=col_prob)
        self.C = A[:, col_indices]
        self.R = A[row_indices, :]
        # Scale the matrices
        self.C = self.C/np.sqrt(r * col_prob[col_indices])
        self.R = self.R/(np.sqrt(r * row_prob[row_indices]).reshape(-1, 1))

        W = A[row_indices, :][:, col_indices]

        scale = np.sqrt(
            r * row_prob[row_indices]).reshape(-1, 1) * np.sqrt(r * col_prob[col_indices])
        W = W/scale

        return self.C, self.R, W

    def decompose(self):
        C, R, W = self.get_C_R_W(int(self.A.shape[0]/10))
        U = np.linalg.pinv(W)
        self.U = U
        self.cur_approx = np.matmul(np.matmul(C, U), R)
        return self.cur_approx

    def retain_90(self):
        middle_diagonal = np.diag(self.W)
        sorted_indices = middle_diagonal.argsort()[::-1]
        total_sum = middle_diagonal.sum()
        sum = 0
        taken = 0
        for i in sorted_indices:
            taken += 1
            sum += middle_diagonal[i]
            if ((sum/total_sum) >= 0.9):
                break
        indices_to_take = sorted_indices[:taken]
        self.C = self.C[:, indices_to_take]
        self.R = self.R[indices_to_take, :]
        middle_diagonal = middle_diagonal[indices_to_take]
        self.W = np.diag(middle_diagonal)


ds = Dataset()
matrix = ds.matrix_np

start = timer()
cur = CUR(matrix)
cur.decompose()
end = timer()
print("Time taken to decompose CUR=", (end-start))
cur_approx = cur.cur_approx
print("Matrix", matrix)
print(cur_approx.shape)
print(cur_approx)
print(cur_approx[-1].mean())

ds.print_metrics(cur_approx)

# print("U", cur.U)
# print("Multiplied is", cur_approx)

# print("Mean squared error with training data")
# print(np.sqrt(np.sum((cur_approx - matrix)**2,
#       where=matrix != 0) / np.count_nonzero(matrix)))

# print("Total non zero", np.count_nonzero(matrix))
# Convert to dataframe
# cur_approx_df = pd.DataFrame(cur_approx, columns=ds.movies_map)
# cur_approx_df.to_csv("dataset/cur_approx.csv")
