import numpy as np
import random
import jieba
import json

from gensim.models import word2vec

empty_vector = []
for i in range(0,100):
    empty_vector.append(float(0.0))
one_vector = []
for i in range(0,10):
    one_vector.append(float(1))
zero_vector = []
for i in range(0,10):
    zero_vector.append(float(0))

def get_rand_index(min, max, exclude):
    while True:
        index = random.randint(min, max)
        if index != exclude:
            return index

def get_vocab():
    # load word vector model
    model_path = './data/word2vec.model'
    model = word2vec.Word2Vec.load(model_path)
    vocab = {}
    for k,v in model.wv.vocab.items():
        vocab[k] = model[k]
    return vocab

def get_train_data():
    train = json.load(open('./data/train.json'))
    return train

def get_test_data():
    test = json.load(open('./data/test.json'))
    return test

def print_random_qa():
    # 一次打印10个qa
    train_data = get_train_data()

    count = 10
    index = random.randint(0, len(train_data) - 1 - count)
    for i in range(index, index + count):
        print('[Q]:' + train_data[i]['q'])
        print('[A]:' + train_data[i]['a'])
        print('----------')

def encode_sent(vocab, string, size):
    # vocab 词典和词向量组成
    # string 句子
    # size 句子长度限制
    result = []
    cutted = jieba.lcut(string)
    
    valid_size = size
    if size > len(cutted):
        valid_size = len(cutted)

    for i in range(0, valid_size):
        if cutted[i] in vocab:
            result.append(vocab[cutted[i]])
        else:
            result.append(empty_vector)

    if valid_size < size:
        for i in range(valid_size, size):
            result.append(empty_vector)

    return result

def load_train_set(vocab, train_data, batch):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, batch):
        qid = random.randint(0, len(train_data) - 1)
        nid = get_rand_index(0, len(train_data) - 1, qid)

        q = encode_sent(vocab, train_data[qid]['q'], 200)
        a = encode_sent(vocab, train_data[qid]['a'], 200)
        n = encode_sent(vocab, train_data[nid]['a'], 200)
        x_train_1.append(q)
        x_train_2.append(a)
        x_train_3.append(n)
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_test_set(vocab, test_data, i, batch = 100):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    
    batch_begin = i * batch
    batch_end = batch_begin + batch
    #print(test_data[batch_begin]['r'])
    for j in range(batch_begin, batch_end):
        q = encode_sent(vocab, test_data[j]['q'], 200)
        a = encode_sent(vocab, test_data[j]['a'], 200)
        
        x_train_1.append(q)
        x_train_2.append(a)
        x_train_3.append(a)
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

#vocab = get_vocab()
#train = get_train_data()
#print(len(train))
#x_1,x_2,x_3 = load_train_set(vocab, train, 10)
#print(np.shape(x_1[0]))
#print(train_set)
#test = get_test_data()
#print(len(test))
#test_set = load_test_set(vocab, test, 1)
#print(test_set[0:1])
#print_random_qa()
