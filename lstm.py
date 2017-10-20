# -*- coding:utf8 -*-
"""
LSTM实现购物评论情感分类
"""
import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential,load_model
from keras.layers import Dense, Embedding
from keras.layers import LSTM

pos_file = './data/pos.xls'
neg_file = './data/neg.xls'
embedding_size = 100 ## 词向量维度

max_features = 20000
maxlen = 100
batch_size = 32


def load_data():
    ## 从文件读取数据
    pos_df = pd.read_excel(pos_file, header=None, index_col=None)
    neg_df = pd.read_excel(neg_file, header=None, index_col=None)

    x = np.concatenate((pos_df[0], neg_df[0]))
    pos_label = np.ones(len(pos_df))
    neg_label = np.zeros(len(neg_df))

    y = np.concatenate((pos_label, neg_label))

    return x, y


def cut_words(text):
    return np.array([list(jieba.cut(sentence.strip())) for sentence in text])


## 训练词向量
def build_word2vec(dataset):
    w2v_model = Word2Vec(size=embedding_size, min_count=5)
    w2v_model.build_vocab(dataset)
    w2v_model.train(dataset, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
    w2v_model.save('./model/lstm_model/w2v_model.pkl')
    return w2v_model


## 每个词对应的下标和词向量
def creat_word_dict(model, dataset):
    if model is not None and dataset is not None:
        w2idx = dict()
        for k, v in model.wv.vocab.items():
            w2idx[k] = v.index+1

        def parse_data(dataset):

            data = []
            for document in dataset:
                new_doc = []
                for word in document:
                    if word in w2idx:
                        new_doc.append(w2idx[word])
                    else:
                        new_doc.append(0)
                data.append(new_doc)
            return data
        dataset = parse_data(dataset)
        dataset = sequence.pad_sequences(dataset, maxlen=maxlen)
        return w2idx, dataset
    else:
        print 'error!'


def gen_data(model, w2idx, dataset, label):

    nb_words = len(model.wv.vocab.keys())+1 ## 所有频数小于5的词索引为0，所以要加1
    weights = np.zeros((nb_words, embedding_size))
    for word, index in w2idx.items():
        weights[index, :] = model[word]
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3)

    return nb_words, weights, X_train, y_train, X_test, y_test


def lstm_model(nb_words, embedding_weights, X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Embedding(input_dim=nb_words, output_dim=embedding_size, weights=[embedding_weights]))
    model.add(LSTM(64, dropout=0.5, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=15,
              verbose=1,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)
    model.save('./model/lstm_model/lstm.h5')
    print('Test score:', score)
    print('Test accuracy:', acc)


def input_transform(text):
    cut_text = jieba.lcut(text)
    cut_text = np.array(cut_text).reshape(1, -1)
    w2v_model = Word2Vec.load('./model/lstm_model/w2v_model.pkl')
    _, test = creat_word_dict(w2v_model, cut_text)
    return test


def lstm_predict(text):
    model = load_model('./model/lstm_model/lstm_model.h5')
    test = input_transform(text)

    pred = model.predict(test)

    if int(pred[0][0] == 1):
        print 'positive'
    else:
        print 'negative'


def train():
    print 'loading data...'
    text, label = load_data()
    print 'cutting text...'
    dataset = cut_words(text)
    print 'word embedding training...'
    w2v_model = build_word2vec(dataset)
    print 'building word index...'
    w2idx, dataset = creat_word_dict(w2v_model, dataset)
    print 'preparing data...'
    nb_words, weights, X_train, y_train, X_test, y_test = gen_data(w2v_model, w2idx, dataset, label)
    np.save('./data/lstm_data/embedding_weights.npy', weights)
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape
    lstm_model(nb_words, weights, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    train()

    string = '昨天下的单，今天就到了，快递真是给力啊，希望产品也一样给力。'
    string = '这是我买的最后悔的一件东西了'
    string = '颜色很亮，无色差，大小合适，先试着看看'

    lstm_predict(string)





