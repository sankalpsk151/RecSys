import pandas as pd
import numpy as np
import cur

# ratings = pd.io.parsers.read_csv('dataset/ratings.dat', names=[
#                                               'userid', 'movieid', 'ratings', 'time'], encoding='latin-1', engine='python', delimiter='::')
# ratings = ratings.drop('time', axis=1)
# ratings.sort_values(by=['userid', 'ratings'])
# userid = 2
# userid_ratings = ratings[ratings['userid'] == userid].reset_index(drop=True).sort_values(by=['ratings'])
# print(userid_ratings.head(129))
# top_k  = 5
# top_kth_rating = userid_ratings.at[top_k-1, 'ratings']
# val3 = userid_ratings[userid_ratings['ratings'] == top_kth_rating].reset_index(drop=True)
# print(val3)
# s1 = set(userid_ratings['movieid'].iloc[:top_k])
# set_of_movies = s1.union(set(val3['movieid']))
# print(set_of_movies)

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
    r = 3

    # Get the row and column probabilities
    A_sq = A**2
    row_prob = np.sum(A_sq, axis=1)
    row_prob = row_prob/np.sum(row_prob)
    col_prob = np.sum(A_sq, axis=0)
    col_prob = col_prob/np.sum(col_prob)

    # Get the row and column indices
    row_indices = np.random.choice(m, r, replace=False, p=row_prob)
    col_indices = np.random.choice(n, r, replace=False, p=col_prob)
    # Get the row and column matrices
    # R = np.take(A, row_indices, axis=0)
    R = A[row_indices, :]
    C = A[:, col_indices]
    # C = np.take(A, col_indices, axis=1)
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

# print(CUR_decomposition(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
c, u, r = CUR_decomposition(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
# u = np.diag(u)
# print((c * u[...,None, :])@r)
print(np.dot(c, np.dot(u, r.T)))

c, u, r = cur.cur_decomposition(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 3)
# u = np.diag(u)
print(np.dot(c, np.dot(u, r.T)))

