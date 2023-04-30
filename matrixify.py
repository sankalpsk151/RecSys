import pandas as pd
import numpy as np


def CUR_decomposition(A):
    """
    A = CUR
    A: m x n matrix
    C: m x r matrix
    U: r x r matrix
    R: r x n matrix
    """
    # Get the shape of the matrix
    m, n = A.shape
    # Get the number of rows and columns
    r = int(m/10)

    # Get the row and column probabilities
    row_prob = np.sum(A**2, axis=1)
    row_prob = row_prob/np.sum(row_prob)
    col_prob = np.sum(A**2, axis=0)
    col_prob = col_prob/np.sum(col_prob)

    # Get the row and column indices
    row_indices = np.random.choice(m, r, replace=False, p=row_prob)
    col_indices = np.random.choice(n, r, replace=False, p=col_prob)
    # Get the row and column matrices
    R = A[row_indices, :]
    C = A[:, col_indices]
    W = A[row_indices, :][:, col_indices]

    print("This is R")
    print(R)
    print("This is C")
    print(C)
    X, Z, Y = np.linalg.svd(W)
    # Take reciprocal of non zero values in Z
    Z = 1/Z

    # Make Z diagonal
    Z = np.diag(Z)
    # print(Z)
    print(X.shape, Z.shape, Y.shape)
    U = Y.T @ (Z**2) @ X.T
    print(U.shape)
    return C, U, R


# train_matrix = pd.read_csv("dataset/matrix.csv")
# train_matrix = train_matrix.drop('Unnamed: 0', axis=1)
# print(train_matrix.head())

# Convert to numpy array
# train_matrix = train_matrix.values
# A = np.array(train_matrix)
# C, U, R = CUR_decomposition(A)
# prediction_matrix = C @ U @ R
# # print(prediction_matrix)
# # Save the prediction matrix
# np.savetxt("dataset/prediction_matrix.csv", prediction_matrix, delimiter=",")
# print(prediction_matrix)

# df = pd.io.parsers.read_csv('dataset/ratings.dat', names=[
#                             'userid', 'movieid', 'ratings', 'time'], encoding='latin-1', engine='python', delimiter='::')
# train_df = pd.read_csv("dataset/train_ratings.csv")
# unique_users = df['userid'].unique()
# unique_movies = df['movieid'].unique()
# print(len(unique_users), len(unique_movies))
# unique_users = train_df['userid'].unique()
# unique_movies = train_df['movieid'].unique()
# print(len(unique_users), len(unique_movies))


def create_matrix(filepath="dataset/train_ratings.csv", dest="dataset/matrix.csv"):
    df = pd.io.parsers.read_csv('dataset/ratings.dat', names=[
                                'userid', 'movieid', 'ratings', 'time'], encoding='latin-1', engine='python', delimiter='::')
    train_df = pd.read_csv(filepath)
    unique_users = df['userid'].unique()
    unique_movies = df['movieid'].unique()

    # Create a matrix
    matrix = pd.DataFrame(index=unique_users, columns=unique_movies)
    # Fill the matrix with the ratings
    for i in range(len(train_df)):
        matrix.loc[train_df['userid'][i],
                   train_df['movieid'][i]] = train_df['ratings'][i]
    # Fill the NaN values with 0
    matrix.fillna(value=0, inplace=True)
    # Save the matrix
    matrix.to_csv(dest)
    print(matrix)

    # 6040, 3685
create_matrix("dataset/test_ratings.csv", "dataset/matrix_test.csv")
