## 文本情感分析

购物评论语料情感分析

### 目录结构

        
    |_
    | |_ data
    | | |_ lstm_data                //保存LSTM模型下的数据
    | | | |_ embedding_weights.npy  //词向量
    | | |_ neg.xls                  //正样本
    | | |_ pos.xls                  //负样本
    | | |_ svm_data                 //保存SVM模型下的数据
    | | | |_ test_label.npy         //验证数据类别
    | | | |_ test_vec.npy           //验证数据
    | | | |_ train_label.npy        //训练数据类别
    | | | |_ train_vec.npy          //训练数据
    | |_ images                     //运行结果截图
    | |_ lstm.py                    //LSTM模型
    | |_ model
    | | |_ lstm_model
    | | | |_ w2v_model.pkl
    | | |_ svm_model
    | | | |_ svm.pkl
    | | | |_ w2v_model.pkl
    | |_ README.md
    | |_ svm.py                     //SVM模型

### SVM实现

embedding_size为词向量维度，这里取100

模型输入是（sample_numbers,embedding_size），对于每一个文本，求所有的词向量均值，
这样一个100维的向量表示一个文本。

划分训练集:测试集 = 7：3

测试结果accuracy=81%

### LSTM实现

embedding_size为词向量维度，这里取100

maxlen = 100 表示一个文本的最大长度

nb_words表示语料库中的词的个数

模型第一层为Embeding层，输入shape=(samples_numbers,nb_words)，输出shape=(sample_numbers,maxlen,embedding_size)

训练15轮，batch_size = 32,最后验证集上accuracy=90.7%


