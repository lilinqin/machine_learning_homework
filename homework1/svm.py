import numpy as np


class SVM():
    def __init__(self, C, max_epoch ):
        self.C = C # 惩罚系数
        self.max_epoch = max_epoch # 最大迭代次数
        self.alpha = None # 拉格朗日乘子
        self.b = 0 # 偏差
        self.Ek = None # 误差缓存表
        self.X = None # 训练数据
        self.Y = None # 标签
        self.N = None # 数据集大小

    # 线性核函数
    def kernel(self, xi, xj): 
        if len(xj.shape) == 1: # 判断是整个数据集还是单个数据
            return np.dot(xi,xj)
        else:
            K = np.zeros(self.N)
            for i in range(self.N):
                K[i] = np.dot(xi, xj[i])
            return K

        
    # 寻找各类点
    def get_index_list(self):
        # 外层循环首先遍历0<a<C的样本，查看是否满足KKT条件，否则遍历全部样本
        index_list = [i for i in range(self.N) if 0 < self.alpha[i] < self.C]
        others = [i for i in range(self.N) if i not in index_list]
        index_list.extend(others)
        return index_list
    
    # 满足KKT条件
    def satisfy_KKT(self, i):
        ygxi = self.Y[i] * self.g(i)
        if 0 < self.alpha[i] < self.C:
            return ygxi == 1
        elif self.alpha[i] == 0:
            return ygxi >= 1
        else:
            return ygxi <= 1
    
    # 计算g（xi)
    def g(self, i):
        res = self.b
        res += np.dot(self.alpha, self.Y * self.kernel(self.X[i], self.X))
        return res


    # 计算误差EK
    def E(self, i):
        return self.g(i) - self.Y[i]

    
    # 选择i和j
    def select_i_j(self, index_list):
        for i in index_list:
            if self.satisfy_KKT(i):
                continue
            
            # 选择|Ei-Ej|最大的j
            Ei = self.Ek[i]
            j = 0
            maxDelta = -1
            for k in range(self.N):
                Ek = self.Ek[k]
                if abs(Ei - Ek) > maxDelta:
                    maxDelta = abs(Ei - Ek)
                    j = k
            
            return i, j # 找到返回i,j
        return None # 找不到返回None

    
    def fit(self, X, Y):
        self.X = X
        self.N = X.shape[0] # 样本数量
        self.Y = Y
        self.b = 0
        self.alpha = np.zeros(self.N) # 拉格朗日乘子
        self.Ek = [self.E(i) for i in range(self.N)] # 误差表

        for k in range(self.max_epoch):
            index_list = self.get_index_list() 
            idxs = self.select_i_j(index_list) # 选择alphai和alphaj
            if not idxs: # 如果找不到更新点结束训练
                break
            i, j = idxs
            
            # 旧值
            alphaiold = self.alpha[i]
            alphajold = self.alpha[j]

            # 获取alphajnew的上下界
            if(self.Y[i] != self.Y[j]):
                L = max(0, alphajold - alphaiold)
                H = min(self.C, self.C + alphajold - alphaiold)
            else:
                L=max(0, alphajold + alphaiold - self.C)
                H=min(self.C, alphajold + alphaiold)
            
            # 获取Ek
            Ei = self.Ek[i]
            Ej = self.Ek[j]

            # 计算Kii+Kjj-2*Kij
            eta = self.kernel(self.X[i], self.X[i]) + self.kernel(self.X[j], self.X[j]) - 2 * self.kernel(self.X[i], self.X[j])
            if eta <= 0:
                continue
            
            alphajnew = alphajold + self.Y[j] * (Ei - Ej) / eta
            # 对alphajnew进行裁剪
            if alphajnew > H:
                alphajnew = H
            elif alphajnew < L:
                alphajnew = L
            else:
                alphajnew = alphajnew

            # 获取alphainew
            alphainew = alphaiold + self.Y[i] * self.Y[j] * (alphajold - alphajnew)

            xi, xj = self.X[i], self.X[j]
            bi = -Ei - self.Y[i] * self.kernel(xi, xi) * (alphainew - alphaiold) -\
                self.Y[j] * self.kernel(xj, xi) * (alphajnew - alphajold) + self.b
            
            bj = -Ej - self.Y[i] * self.kernel(xi, xj) * (alphainew - alphaiold) -\
                self.Y[j] * self.kernel(xj, xj) * (alphajnew - alphajold) + self.b
            
            # 更新b
            if 0 < alphainew < self.C: 
                self.b = bi 
            elif 0 < alphajnew < self.C: 
                self.b = bj
            else: 
                self.b = 0.5 * (bi + bj)
            
            # 更新alpha
            self.alpha[i] = alphainew
            self.alpha[j] = alphajnew
            
            # 更新Ek
            self.Ek[i] = self.E(i)
            self.Ek[j] = self.E(j)
        
        # print('训练完成')


    def predict(self, X_test):
        n = X_test.shape[0]
        res_list = np.zeros(n)
        for i in range(n): 
            res = np.sum(self.alpha * self.Y * self.kernel(X_test[i], self.X))
            res += self.b
            res_list[i] = res
        res_list = [1 if i > 0 else -1 for i in res_list]
        return res_list
        



