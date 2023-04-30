from timeit import default_timer as timer
import pandas as pd
import numpy as np
from util import spearman_with_ties


class Dataset:
    """Holds the training and testing datasets"""

    def __init__(self):
        self.original_df = pd.io.parsers.read_csv('dataset/ratings.dat', names=[
                                                  'userid', 'movieid', 'ratings', 'time'], encoding='latin-1', engine='python', delimiter='::')
        self.get_train_test_df()
        self.get_train_matrix()
        self.get_test_matrix()

    def get_train_test_df(self):
        """Read the training and testing CSVs as Pandas Dataframe"""

        self.train_df = pd.read_csv("dataset/train_ratings.csv")
        self.test_df = pd.read_csv("dataset/test_ratings.csv")
        self.train_df = self.train_df.drop('Unnamed: 0', axis=1)
        self.test_df = self.test_df.drop('Unnamed: 0', axis=1)
        return self.train_df, self.test_df

    def get_train_matrix(self):
        """Read the matrix.csv to a numpy matrix"""

        self.matrix = pd.read_csv("dataset/matrix_train.csv")
        self.matrix = self.matrix.drop('Unnamed: 0', axis=1)
        self.matrix.columns = self.matrix.columns.astype(int)
        self.movies_map = self.matrix.columns
        self.matrix_np = self.matrix.to_numpy()
        
        # To handle generous raters
        self.bool_mat = np.where(self.matrix_np == 0, False, True)
        self.user_deviation = np.mean(self.matrix_np, where=self.bool_mat, axis=1)
        self.user_deviation = np.reshape(self.user_deviation, (self.user_deviation.shape[0], 1))
        self.matrix_np = np.subtract(self.matrix_np, self.user_deviation, where=self.bool_mat)

        self.movies_d = {}
        for i in range(len(self.movies_map)):
            self.movies_d[self.movies_map[i]] = i
        return self.matrix_np, self.movies_d
    
    def get_test_matrix(self):
        self.matrix_test = pd.read_csv("dataset/matrix_test.csv")
        self.matrix_test = self.matrix_test.drop('Unnamed: 0', axis=1)
        self.matrix_test.columns = self.matrix_test.columns.astype(int)
        self.movies_map_test = self.matrix_test.columns
        self.matrix_np_test = self.matrix_test.to_numpy()

        self.bool_mat_test = np.where(self.matrix_np_test == 0, False, True)
        self.matrix_np_test = np.subtract(self.matrix_np_test, self.user_deviation, where=self.bool_mat)
        
        self.movies_d = {}
        for i in range(len(self.movies_map)):
            self.movies_d[self.movies_map[i]] = i
        return self.matrix_np, self.movies_d

    def RMSE_training(self, predictionMatrix):
        """RMSE on training
        Parameters:
        predictionMatrix (numpy matrix): The predicted matrix. Must be the same shape as matrix_np
        """
        return np.sqrt(((self.matrix_np - predictionMatrix)**2).sum(where=self.matrix_np != 0) / np.count_nonzero(self.matrix_np))

    def RMSE_testing(self, predictionMatrix):
        """RMSE on testing dataset

        Parameters:
        predictionMatrix (numpy matrix): The predicted matrix. Must be the same shape as matrix_np
        """
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
            predicted_rating = predictionMatrix[user_index][movie_index] + self.user_deviation[user_index]
            # Calculate the error
            error = (rating - predicted_rating)**2
            # Add the error to the total error
            total_error += error
        ans = np.sqrt(total_error / len(self.test_df))
        return ans[0]

    def print_metrics(self, predictionMatrix):
        """
        Calculate the training and testing RMSE, Spearman Coefficient and Precision
        at top 4

        Parameters:
        predictionMatrix (numpy matrix): The predicted matrix. Must be the same shape as matrix_np

        """
        print(f"Training RMSE: {self.RMSE_training(predictionMatrix)}")
        print(f"Testing RMSE: {self.RMSE_testing(predictionMatrix)}")
        print(f"Train Spearman Coefficient: {self.get_spearman_coef(predictionMatrix, self.matrix_np)}")
        print(f"Test Spearman Coefficient: {self.get_spearman_coef(predictionMatrix, self.matrix_np_test)}")
        print(f"Train Precision at top 4: {self.precision_at_top_k(4, predictionMatrix, self.matrix_np)}")
        print(f"Test Precision at top 4: {self.precision_at_top_k(4, predictionMatrix, self.matrix_np_test)}")

    def get_spearman_coef(self, predictionMatrix, matrix_np):
        """Calculate the average spearman coefficient over all users

        Parameters:
        predictionMatrix (numpy matrix): The predicted matrix. Must be the same shape as matrix_np

        """

        total = 0
        for i in range((matrix_np.shape[0])):
            total += spearman_with_ties(matrix_np[i], predictionMatrix[i])
        return total/matrix_np.shape[0]

    def precision_at_top_k(self, k, predictionMatrix, matrix_np):
        """
        Calculate precision at top k averaging over all users

        Parameters:
        predictionMatrix (numpy matrix): The predicted matrix. Must be the same shape as matrix_np

        """
        def is_relevant(rating, row_index):
            return rating >= 3 - self.user_deviation[row_index]

        correct_preds = 0
        for row_index in range(matrix_np.shape[0]):
            local_correct = 0
            sorted_indices = matrix_np[row_index].argsort()[::-1]
            for i in range(k):
                movie_index = sorted_indices[i]
                if is_relevant(predictionMatrix[row_index][movie_index], row_index):
                    local_correct += 1
            correct_preds += local_correct/k

        return correct_preds/matrix_np.shape[0]


class SVD:
    """Class for SVD decomposition"""

    def __init__(self, matrix):
        self.matrix = matrix
        
    def decompose(self):
        """Decomposes the matrix into U, sigma and V_T"""
        return np.linalg.svd(self.matrix, full_matrices=False)

    def decompose90(self):
        """Decompose the matrix into U, sigma and V_T by retaining 90% diagonal values"""
        U, sigma, V_T = np.linalg.svd(self.matrix, full_matrices=False)
        middle_diagonal = (sigma)
        # print(middle_diagonal)
        sorted_indices = middle_diagonal.argsort()[::-1]
        # print(sorted_indices)
        total_sum = (middle_diagonal**2).sum()
        sum = 0
        taken = 0
        for i in sorted_indices:
            taken += 1
            sum += middle_diagonal[i]**2
            if ((sum/total_sum) >= 0.9):
                break
        indices_to_take = sorted_indices[:taken]
        U = U[:, indices_to_take]
        V_T = V_T[indices_to_take, :]
        middle_diagonal = middle_diagonal[indices_to_take]
        sigma = (middle_diagonal)
        return U, sigma, V_T

    def get_predictions(self, decomposition):
        """Multiply given decomposition to get the approximate matrix"""
        U, sigma, V_T = decomposition
        return np.dot(U, np.dot(np.diag(sigma), V_T))


class CUR:
    """Class for CUR decomposition"""

    def __init__(self, matrix):
        self.A = matrix
        self.matrix = matrix
        self.C = None
        self.U = None
        self.R = None
        self.cur_approx = None

    def get_probabilities(self):
        """Calcualtes the probabilities of selecting rows and columns

        Returns:
        row and column probabilities
        """
        A = self.A
        row_prob = np.sum(A**2, axis=1)
        row_prob = row_prob/np.sum(row_prob)
        col_prob = np.sum(A**2, axis=0)
        col_prob = col_prob/np.sum(col_prob)
        return row_prob, col_prob

    def get_C_R_W(self, r):
        """
        Construct C, R, W matrices

        Parameters:
        r (int) : Number of columns and rows to select randomnly

        Returns:
        C : r randomly picked columns and scaled according to probabilities 
        R : r randomly picked rows and scaled according to probabilities
        W : Intersection of C and R
        """
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
        """Performs CUR decomposition
        Returns: 
        Approximate matrix after decomposition
        """
        # C, R, W = self.get_C_R_W(int(self.A.shape[0]/10))
        C, R, W = self.get_C_R_W(3000)

        # C, R, W = self.get_C_R_W(100)

        U = np.linalg.pinv(W)
        self.U = U
        self.cur_approx = C @ U @ R  # np.matmul(np.matmul(C, U), R)
        return self.cur_approx

    def decompose90(self):
        """Performs CUR decomposition by retaining 90% energy
        Returns: 
        Approximate matrix after decomposition
        """
        C, R, W = self.get_C_R_W(int(0.9*3000))

        U = np.linalg.pinv(W)
        self.U = U
        # print(C.shape, U.shape, R.shape)
        self.cur_approx = C @ U @ R
        # self.cur_approx = np.matmul(np.matmul(C, U), R)
        return self.cur_approx


if __name__ == '__main__':
    ds = Dataset()
    matrix = ds.matrix_np

    start = timer()
    cur = CUR(matrix)
    cur.decompose()
    end = timer()
    cur_approx = cur.cur_approx

    print("-----CUR Normal-----")
    print("Time: ", (end-start))
    ds.print_metrics(cur_approx)

    print("-----CUR 90%-----")
    t0 = timer()
    decompose_90 = cur.decompose90()
    print("Time: ", timer() - t0)
    ds.print_metrics(decompose_90)

    print("-----SVD------")
    svd = SVD(cur.A)
    t0 = timer()
    svd_approx = svd.get_predictions(svd.decompose())
    print("Time: ", timer() - t0)

    ds.print_metrics(svd_approx)
    print("-----SVD 90%------")
    t0 = timer()
    svd_90_approx = svd.get_predictions(svd.decompose90())
    print("Time: ", timer() - t0)

    ds.print_metrics(svd_90_approx)
