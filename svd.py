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
        self.ans = (self.U * self.S[..., None, :]) @ self.V
        print(f"Time taken to do Multiplication: {time() - t0} seconds")
        self.get_results()
        
    def get_results(self):
        t0 = time()
        self.pred_train = [self.ans[i-1][j-1] for i, j in zip(
            self.train['userid'], self.train['movieid'])]
        print(
            f"Time taken to predict Train: {time() - t0} seconds")

        t0 = time()
        self.pred_test = [self.ans[i-1][j-1] for i, j in zip(
            self.test['userid'], self.test['movieid'])]
        print(
            f"Time taken to predict Test: {time() - t0} seconds")

class SVDRetention(SVD):
    def __init__(self, U, V, S, train, test, k) -> None:
        self.U = U
        self.V = V
        self.S = S
        self.test = test
        self.train = train
        self.retain_k(k)
        t0 = time()
        self.ans = (self.U * self.S[..., None, :]) @ self.V
        print(f"Time taken to do Multiplication: {time() - t0} seconds")
        self.get_results()
        
        
    def retain_k(self, k):
        total = np.sum(np.square(self.S))*k
        rf = len(self.S)
        for i in range(0,len(self.S)):
            if np.sum(np.square(self.S[:i])) > total:
                print(f"Retained {i} features")
                rf = i
                break
        self.U = self.U[:, :rf]
        self.S = self.S[:rf]
        self.V = self.V[:rf, :]


if __name__ == "__main__":
    data = Dataset()
    ev = EvaluttionMetrics()
    
    print("======== SVD =========")
    svd = SVD(data.matrix, data.train_ratings, data.test_ratings)
    print(
    f"Training RMSE: {ev.get_RMSE(svd.train['ratings'], svd.pred_train)}")
    print(
    f"Test RMSE : {ev.get_RMSE(svd.test['ratings'], svd.pred_test)}")
    
    print("======== SVD with Retention =========")
    svd_ret = SVDRetention(svd.U, svd.V, svd.S, data.train_ratings, data.test_ratings, 0.9)
    print(
    f"Training RMSE : {ev.get_RMSE(svd_ret.train['ratings'], svd_ret.pred_train)}")
    print(
    f"Test RMSE : {ev.get_RMSE(svd_ret.test['ratings'], svd_ret.pred_test)}")
    