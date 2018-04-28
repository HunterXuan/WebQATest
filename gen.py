from gensim.models import word2vec
import json
import random

model_path = './data/word2vec.model'
model = word2vec.Word2Vec.load(model_path)

webqa = json.load(open('./data/webqa.json'))

def get_rand_index(min, max, exclude):
    while True:
        index = random.randint(min, max)
        if index != exclude:
            return index

def get_rand_answer(webqa, exclude):
    index = get_rand_index(0, len(webqa) - 1, exclude)
    return webqa[index]['a']

train = []
i = 0
for id,qa in enumerate(webqa):
    tmp = {}
    tmp['qid'] = id
    tmp['q'] = qa['q']
    tmp['a'] = qa['a']
    tmp['r'] = 1
    train.append(tmp)
    #for j in range(1,10):
    #    tmp['a'] = get_rand_answer(webqa, id)
    #    tmp['r'] = 0
    #    train.append(tmp)

json.dump(train, open('./data/train.json', 'w'))

test = []
i = 0
for id,qa in enumerate(webqa):
    tmp = {}
    tmp['qid'] = id
    tmp['q'] = qa['q']
    tmp['a'] = qa['a']
    tmp['r'] = 1
    test.append(tmp)
    for j in range(1,100):
        tmp = {}
        tmp['qid'] = id
        tmp['q'] = qa['q']
        tmp['a'] = get_rand_answer(webqa, id)
        tmp['r'] = 0
        test.append(tmp)

json.dump(test, open('./data/test.json', 'w'))
