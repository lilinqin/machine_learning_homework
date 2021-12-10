import numpy as np

class LogisticRegression():
    def __init__(self, feature_num=4, max_epoch=100, lr=0.1):
        self.w = np.zeros(feature_num)
        self.b = 0
        self.max_epoch = max_epoch
        self.lr = lr
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, Y):
        n = X.shape[0]
        for _ in range(self.max_epoch):
            for i in range(n):
                z = np.dot(self.w, X[i]) + self.b
                y_hat = self.sigmoid(z)

                self.w += self.lr * (Y[i] - y_hat) * X[i]
                self.b += self.lr * (Y[i] - y_hat)

    
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        res_list = [1 if i > 0 else -1 for i in z]
        return res_list

