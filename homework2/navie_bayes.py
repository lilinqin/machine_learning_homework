import re
import string
import numpy as np
from sklearn.metrics import classification_report

# 获取数据和词表
def get_data_and_voc(path):
    pattern = string.punctuation
    with open(path, 'r', encoding='unicode_escape') as f:
        i = 0
        texts, labels = [], []
        voc = set()
        for line in f.readlines():
            if i % 2 == 0:
                labels.append(int(line.split('.')[0]))
            else:
                line = line.strip().lower()
                line = re.sub('[{}]'.format(pattern), ' ', line) # 去除标点符号
                line = line.split()
                voc |= set(line)
                texts.append(line)
            i += 1
        voc = list(voc)
        word2id = {v: k for k, v in enumerate(voc)}
    return texts, labels, word2id

# 将文本中每个单词变成词向量
def word2vec(word2id, text):
    return_vec = [0] * len(word2id)
    for word in text:
        return_vec[word2id[word]] += 1
    return return_vec


class Bayes:
    def __init__(self):
        self.p = None # 每个类别的每个单词的概率
        self.p_labels = None # 每个类别的概率

    # 训练
    def fit(self, texts, labels, word2id):
        num_text = len(texts)
        label_set = set(labels)
        num_labels = np.zeros(len(label_set))
        # 计算每个类别概率
        for i in range(len(label_set)):
            num_labels[i] = labels.count(i) / num_text
        self.p_labels = np.log(num_labels)

        p_num = np.ones((len(label_set), len(word2id)))
        p_all = np.ones((len(label_set),1)) + len(label_set)

        # 计算每个类别中每个单词的个数
        for i in range(num_text):
            p_num[labels[i]] += texts[i]
            p_all[labels[i]] += sum(texts[i])
        
        self.p = np.log(p_num / p_all)
    
    # 预测
    def predict(self, test_texts):
        res = []
        for i in range(len(test_texts)):
            logp = np.sum(test_texts[i] * self.p, axis=1)
            logp = logp.reshape((-1))
            logp += self.p_labels
            res.append(logp.argmax())
        return res

        
def main():
    data_path = 'AsianReligionsData.txt'
    texts, labels, word2id = get_data_and_voc(data_path) # 获取数据和词表
    num_text = len(texts)
    # 将文本中每个单词变成词向量
    for i in range(num_text):
        texts[i] = word2vec(word2id, texts[i])

    # 计算每个类别文件个数
    label_set = set(labels)
    num_labels = [0 for _ in range(len(label_set))]
    for i in range(len(label_set)):
        num_labels[i] = labels.count(i)

    # 划分数据集
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    accumulate = 0
    for i in range(len(label_set)):
        split_idx = accumulate + round(num_labels[i] * 0.8) 
        train_texts.extend(texts[accumulate:split_idx])
        train_labels.extend(labels[accumulate:split_idx])
        test_texts.extend(texts[split_idx:accumulate + num_labels[i]])
        test_labels.extend(labels[split_idx:accumulate + num_labels[i]])
        accumulate += num_labels[i]

    train_texts = np.array(train_texts)
    test_texts = np.array(test_texts)

    bayes = Bayes()
    bayes.fit(train_texts, train_labels, word2id) # 训练
    res = bayes.predict(test_texts) # 测试
    evaluate = classification_report(test_labels, res, target_names=list(map(str, range(len(label_set)))), zero_division=False) # 评估
    print(evaluate)

main()