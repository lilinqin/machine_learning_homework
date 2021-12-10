import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm 
import random

def readfile(fname): # 读取文件
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        first = True
        data = [] # 数据，单词组成的列表
        labels = [] # 标签
        id2label, label2id = {}, {} # 标签和索引映射
        id2word, word2id = {}, {} # 单词和索引的映射
        for line in lines:
            if first: # 判断是否是数据内容
                first = False
                continue
            _, line = line.split('\t\t')
            line = line.strip().split()
            word_temp = []
            label_temp = []
            pre = None # 记录[你]这类词
            for word_label in line:
                if '/' in word_label: # 判断是否可以分割
                    if pre: # 如果有[你]这类词，合并
                        word_label = pre + ' ' + word_label
                    word, label = word_label.split('/')
                    pre = None # 清除[你]这类词的记录
                    if label == '': # 若出现 you/ 这种未标注的token，直接去除
                        continue
                else:
                    pre = word_label
                    continue
                word_temp.append(word)
                label_temp.append(label)
                if word not in word2id: # 记录单词及其索引
                    word2id[word] = len(word2id)
                    id2word[len(id2word)] = word 
                if label not in label2id: # 记录标签及其索引
                    label2id[label] = len(label2id)
                    id2label[len(id2label)] = label
            data.append(word_temp)
            labels.append(label_temp)

    
        return id2label, label2id, id2word, word2id, data, labels 

def log(v): # 计算log，保证数值稳定
    if v != 0:
        return np.log(v)
    else:
        return np.log(v + 1e-6)

class HMM():
    def __init__(self, id2label, label2id, id2word, word2id):
        # 词到索引的映射和标签到索引的映射
        self.id2label = id2label 
        self.label2id = label2id
        self.id2word = id2word
        self.word2id = word2id

        # 标签和token个数
        self.label_len = len(id2label)
        self.word_len = len(word2id)

        self.pi = np.zeros(self.label_len) # 初始概率
        self.A = np.zeros((self.label_len, self.label_len)) # 状态转移矩阵
        self.B = np.zeros((self.label_len, self.word_len)) # 观测概率矩阵
    
    # 获取标签对应的索引
    def get_label_id(self, test_labels):
        gold = []
        for labels in test_labels:
            labels = [self.label2id[label] for label in labels]
            gold.extend(labels)
        return gold

    # 训练
    def train(self, train_data, train_labels):
        print('开始训练!')
        for i in tqdm(range(len(train_data))):
            line, labels = train_data[i], train_labels[i]
            for j in range(len(line)):
                word = line[j]
                label = labels[j]

                # 获取对应下标
                word_id = self.word2id[word]
                label_id = self.label2id[label]

                if j == 0: # 如果是第一个词，更新初始概率矩阵
                    self.pi[label_id] += 1
                else: # 否则更新状态转移矩阵
                    self.A[pre_id][label_id] += 1
                self.B[label_id][word_id] += 1 # 更新观测概率矩阵

                pre_id = label_id # 记录前一个标签
        
        # 计算概率
        self.pi = self.pi / sum(self.pi)
        for i in range(len(self.A)):
            self.A[i] /= sum(self.A[i])
            self.B[i] /= sum(self.B[i])

        # 取对数
        for i in range(self.label_len):
            self.pi[i] = log(self.pi[i])
        for i in range(self.label_len):
            for j in range(self.label_len):
                self.A[i][j] = log(self.A[i][j])
        for i in range(self.label_len):
            for j in range(self.word_len):
                self.B[i][j] = log(self.B[i][j])

        print('训练结束!')

    # viterbi算法，预测推理词性
    def viterbi(self, sentence):
        l = len(sentence)
        words = [self.word2id[word] for word in sentence] # 转化为id
        theta = np.zeros((l, self.label_len)) # 在时刻t下状态为i的单个路径中概率最大值
        tabel = np.zeros((l, self.label_len), dtype=int) # 记录在时刻t状态为i的所有单个路径中概率最大的路径的第t-1个结点

        # 初始化
        for i in range(self.label_len):
            theta[0][i] = self.pi[i] + self.B[i][words[0]]  # theta[0][i] = pi[i] * B[i][o1]

        for i in range(1,l): # 时刻i
            for j in range(self.label_len): # 状态j
                theta[i][j] = float('-inf')
                for k in range(self.label_len): # 状态k
                    val = theta[i-1][k] + self.A[k][j] + self.B[j][words[i]] # val = theta[i-1][k] * A[k][j] * B[j][oi]
                    if val > theta[i][j]:
                        theta[i][j] = val # 记录最优值
                        tabel[i][j] = k # 记录最优结点
        
        res = [0 for _ in range(l)] # 最终标签序列
        res[-1] = int(np.argmax(theta[l-1])) # 记录句尾的词性

        for i in range(l-2, -1, -1):
            res[i] = tabel[i+1][res[i+1]] # 查表找出最佳状态路径
        
        return res

    def predict(self, test_data):
        pre = []
        print('开始预测!')
        for i in tqdm(range(len(test_data))):
            res = self.viterbi(test_data[i])
            pre.extend(res)
        print('预测结束!')
        return pre
        
        


fname = 'corpus_你_20211101164534.txt'
id2label, label2id, id2word, word2id, data, labels = readfile(fname) # 读取文件，获取标签、数据等信息

tags = list(sorted(list(label2id))) # 按照字典序排序
label2id = {label:idx for idx,label in enumerate(tags)}
id2label = {v:k for k,v in label2id.items()}


# 按照4:1随机划分训练和测试集
data_len = len(data)
index = list(range(data_len))
random.seed(0)
random.shuffle(index)

train_len = int(data_len * 0.8)
train_data = data[:train_len]
train_labels = labels[:train_len]
test_data = data[train_len:]
test_labels = labels[train_len:]


hmm = HMM(id2label, label2id, id2word, word2id)
hmm.train(train_data, train_labels) # 训练
pre = hmm.predict(test_data) # 预测
gold = hmm.get_label_id(test_labels) # 获取对应的标签id
target_names = [id2label[i] for i in range(len(id2label))] # 获取每个标签
evaluate = classification_report(gold, pre, labels=range(len(target_names)), target_names=target_names, zero_division=False) # 评估
print('======================HMM预测结果======================')
print(evaluate)


