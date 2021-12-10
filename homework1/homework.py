import numpy as np
from sklearn.metrics import classification_report
from perception import Perceptron
from logistic_regression import LogisticRegression
from svm import SVM

# 设置初始随机种子数
np.random.seed(0)

# 读取数据，获取训练集和数据集
def getdata():
    with open('iris.data') as f:
        lines = f.readlines()
        label2idx = {'Iris-setosa': 1, 'Iris-versicolor': -1}
        datas = []
        labels = []

        for line in lines:
            line = line.strip()
            line = line.split(',')
            feature, label = line[:-1], line[-1]
            feature = list(map(float, feature))
            label = label2idx[label]
            datas.append(feature)
            labels.append(label)

        # 划分训练集和数据集
        train_data = datas[:40] + datas[50:90]
        test_data = datas[40:50] + datas[90:]
        train_label = labels[:40] + labels[50:90]
        test_label = labels[40:50] + labels[90:]

        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_label = np.array(train_label)
        test_label = np.array(test_label)

        return train_data, train_label, test_data, test_label

# 改变label，将-1变为0，方便评估计算
def changeLabel(label):
    for i in range(len(label)):
        if label[i] == -1:
            label[i] = 0
    return label
# 标准化
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def main():
    train_data, train_label, test_data, test_label = getdata() # 获取数据
    target_names = ['Iris-versicolor', 'Iris-setosa']
    test_label = changeLabel(test_label) # 改变label，将-1变为0，方便评估计算
    m = train_data.shape[1] # 训练数据特征大小

    # 预处理
    train_data = standardization(train_data)
    test_data = standardization(test_data)

    perceptron = Perceptron(feature_num=m, max_epoch=100, lr=0.1) # 初始化
    perceptron.fit(train_data, train_label) # 训练
    res = perceptron.predict(test_data) # 预测
    y_pred = changeLabel(res) # 改变label，-1 --> 0
    evaluate = classification_report(test_label, y_pred, target_names=target_names, zero_division=False) # 评估
    print('=====================感知机预测结果=====================')
    print(evaluate)

    svm = SVM(C=1.0, max_epoch=1000)
    svm.fit(train_data, train_label)
    res = svm.predict(test_data)
    y_pred = changeLabel(res)
    evaluate = classification_report(test_label, y_pred, target_names=target_names, zero_division=False)
    print('===================支持向量机预测结果===================')
    print(evaluate)


    train_label = changeLabel(train_label) # 修改label，-1 --> 0
    logisticRegression = LogisticRegression(feature_num=m, max_epoch=100, lr=0.1)
    logisticRegression.fit(train_data, train_label)
    res = logisticRegression.predict(test_data)
    y_pred = changeLabel(res)
    evaluate = classification_report(test_label, y_pred, target_names=target_names, zero_division=False)
    print('====================逻辑回归预测结果====================')
    print(evaluate)

main()

