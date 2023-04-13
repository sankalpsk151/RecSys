import numpy as np
from time import time
from util import Dataset, EvaluttionMetrics

class SVD:
    def __init__(self, matrix, train, test) -> None:
        self.matrix = matrix
        self.test = test
        self.train = train
        # self.mean = np.mean(self.matrix)
        t0 = time()
        self.U, self.S, self.V = np.linalg.svd(self.matrix, full_matrices=False)
        print(f"Time taken to do SVD: {time() - t0} seconds")
        print(self.U.shape, self.S.shape, self.V.shape)
    
        t0 = time()
        # self.ans = np.matmul(self.U, np.matmul(np.diag(self.S), self.V.T))
        self.ans = (self.U * self.S[..., None, :]) @ self.V
        print(f"Time taken to do Multiplication: {time() - t0} seconds")
        self.get_results()
        
    # def get_rating(self, userid, movieid):
    #     rating = 
    #     return rating
        
    def get_results(self):
        t0 = time()
        self.pred_train = [self.ans[i-1][j-1] for i, j in zip(
            self.train['userid'], self.train['movieid'])]
        print(
            f"Time taken to predict using SVD Train: {time() - t0} seconds")

        t0 = time()
        self.pred_test = [self.ans[i-1][j-1] for i, j in zip(
            self.test['userid'], self.test['movieid'])]
        print(
            f"Time taken to predict using SVD Test: {time() - t0} seconds")

if __name__ == "__main__":
    data = Dataset()
    ev = EvaluttionMetrics()
    
    print("======== SVD =========")
    svd = SVD(data.matrix, data.train_ratings, data.test_ratings)
    print(
    f"Training RMSE for SVD: {ev.get_RMSE(svd.train['ratings'], svd.pred_train)}")
    print(
    f"Test RMSE for SVD: {ev.get_RMSE(svd.test['ratings'], svd.pred_test)}")
    