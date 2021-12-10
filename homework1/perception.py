import numpy as np

# 设置初始随机种子数
np.random.seed(0)


class Perceptron():
    def __init__(self, feature_num=4, max_epoch=100, lr=0.1, random=False):
        # 是否随机初始化参数
        if random:
            self.w = np.random.uniform(size=feature_num)
            self.b = np.random.uniform()
        else:
            self.w = np.zeros(feature_num)
            self.b = 0
        self.max_epoch = max_epoch # 最大迭代次数
        self.lr = lr # 学习率
    
    def fit(self, features, labels):
        n = len(labels) 
        all_true = True # 判断是否有误分类点

        for i in range(1, self.max_epoch + 1):
            for j in range(n):
                res = labels[j] * (np.dot(self.w, features[j]) + self.b)
                if res <= 0:
                    self.w += self.lr * labels[j] * features[j]
                    self.b += self.lr * labels[j]
                    all_true = False
            if all_true:
                break
            else:
                all_true = True
        # print('训练完成')

    
    def predict(self, features):
        res = np.matmul(features, self.w) + self.b
        res = [1 if x >= 0 else -1 for x in res]
        res = np.array(res)
        return res

            
        


