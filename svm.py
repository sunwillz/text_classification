# -*- coding:utf8 -*-
import os
import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


pos_file = './data/pos.xls'
neg_file = './data/neg.xls'
embedding_size = 100


def load_data(filename):
    return pd.read_excel(filename, header=None, index_col=None)


def cut_word(data_df):
    cw = lambda x: list(jieba.cut(x))
    return data_df[0].apply(cw)


## 将一片文档转化向量表示，采用词向量取均值的方法
def build_vec(doc, model):
    cnt = 0
    vec = np.zeros((1, embedding_size))
    for w in doc:
        try:
            vec += model[w].reshape((1, embedding_size))
            cnt += 1
        except KeyError:
            continue
    if cnt != 0:
        vec /= cnt
    return vec


def data_process():
    pos_data = load_data(pos_file)
    neg_data = load_data(neg_file)

    pos_data = cut_word(pos_data)
    neg_data = cut_word(neg_data)

    x = np.concatenate((pos_data, neg_data))
    y_pos = np.ones(len(pos_data)) ## 1表示正类
    y_neg = np.zeros(len(neg_data)) ## 0表示负类
    y = np.concatenate((y_pos, y_neg))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    w2v_model = Word2Vec(size=embedding_size, min_count=5)
    w2v_model.build_vocab(X_train)
    w2v_model.train(X_train, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)

    train_vec = np.concatenate([build_vec(d, w2v_model) for d in X_train])
    np.save('./data/svm_data/train_vec.npy', train_vec)
    np.save('./data/svm_data/train_label.npy', y_train)
    print train_vec.shape

    w2v_model.train(X_test, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
    w2v_model.save('./model/svm_model/w2v_model.pkl')
    test_vec = np.concatenate([build_vec(d, w2v_model) for d in X_test])
    np.save('./data/svm_data/test_vec.npy', test_vec)
    np.save('./data/svm_data/test_label.npy', y_test)
    print test_vec.shape

    return train_vec, y_train, test_vec, y_test


def svm_train(x, y):
    clf = SVC(kernel='rbf')
    clf.fit(x, y)
    joblib.dump(clf, './model/svm_model/svm.pkl')

    return clf


def svm_predict(X_test):
    x = jieba.cut(X_test)
    w2v_model = Word2Vec.load('./model/svm_model/w2v_model.pkl')
    test_vec = build_vec(x, w2v_model)
    if os.path.exists('./model/svm_model/svm.pkl'):
        clf = joblib.load('./model/svm_model/svm.pkl')
        pred = clf.predict(test_vec)
        if int(pred[0]) == 1:
            print 'positive'
        else:
            print 'negative'
    else:
        print 'error!'

if __name__ == '__main__':
    X_train, y_train, X_test,y_test = data_process()
    clf = svm_train(X_train, y_train)
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print 'SVM classifier accuracy = {}'.format(accuracy)

    str ='这台电脑价格虽然比较高，但用户体验还不错，特别是显示屏，看着很爽！'

    svm_predict(str)
