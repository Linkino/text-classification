
from attention import AttentionWithContext
from data_themedata import TextData
# coding=utf-8   #默认编码格式为utf-8
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
# import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from gensim.models import word2vec
import gensim
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from gensim.test.utils import datapath, get_tmpfile, common_texts
from gensim.corpora import LowCorpus
from gensim.corpora import Dictionary
from keras import optimizers
import re
from sklearn.metrics import classification_report
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import f1_score
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
#数据读取与去除theme列中空值所在行


def load_data():
    sql = "select theme,abstract,tag_one,id from theme_data where tag_one is not null and abstract is not null"
    pTextData = TextData()
    data_rows = pTextData.load_table(sql)
    title_list = []
    abstract_list = []
    y_list = []
    for row in data_rows:
        title = row[0]
        content = row[1]
        y = row[2]
        title_list.append(title)
        abstract_list.append(content)
        y_list.append(y)
    return title_list,abstract_list,y_list
def creat_model(wordindex,wordindex1,matrix0,maxlen0,X_train, X_test, y_train, y_test):
    embedding_layer0 = Embedding(len(wordindex) + len(wordindex1) + 2, 256, weights=[matrix0], input_length=maxlen0)
    main_input0 = Input(shape=(maxlen0,), dtype='float64')
    embed = embedding_layer0(main_input0)
    # embedding_layer1 = Embedding(len(wordindex1) + 1, 256, weights=[embedding_matrix1], input_length=maxlen1)
    # main_input1 = Input(shape=(maxlen1,), dtype='float64')
    # embed1 = embedding_layer1(main_input1)
    # 词嵌入（使用预训练的词向量）
    # embed = concatenate([embed0, embed1], axis=-1)
    # 词窗大小分别为3,4,5
    cnn1 = Convolution1D(100,kernel_size=3 , padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=int(cnn1.shape[1]))(cnn1)
    drop1 = Dropout(0.5)(cnn1)
    cnn1 = Bidirectional(LSTM(256, return_sequences=True))(drop1)

    # att_layer1=AttentionWithContext()(cnn1)
    cnn2 = Convolution1D(100,kernel_size=4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=int(cnn2.shape[1]))(cnn2)
    drop2 = Dropout(0.5)(cnn2)
    cnn2 = Bidirectional(LSTM(256, return_sequences=True))(drop2)
    # att_layer2=AttentionWithContext()(cnn2)
    cnn3 = Convolution1D(100, kernel_size=5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=int(cnn3.shape[1]))(cnn3)
    drop3 = Dropout(0.5)(cnn3)
    cnn3 = Bidirectional(LSTM(256, return_sequences=True))(drop3)
    # att_layer3=AttentionWithContext()(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    drop4 = Dropout(0.5)(cnn)
    # flat = Flatten()(cnn)
    att_layer2 = AttentionWithContext()(drop4)
    main_output = Dense(7, activation='softmax')(att_layer2)
    model = Model(inputs=main_input0, outputs=main_output)
    optimizer = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy',
    # 		optimizer=optimizer,
    # 		metrics=[f1_score])
    model.summary()
    earlystopping = EarlyStopping(monitor='val_acc', min_delta=1e-2, patience=3, verbose=2, mode='auto')
    model.fit(X_train, y_train, verbose=1, batch_size=batch_size, epochs=n_epoch, validation_data=(X_test, y_test),
              callbacks=[earlystopping])
    filepath = "./model/sen_model12_yh_theme.h5"
    model.save(filepath=filepath, include_optimizer=True)
    score, acc = model.evaluate(X_test, y_test, verbose=1, batch_size=batch_size)
    return model

if __name__ == "__main__":
    a=[]
    b=[]
    c=[]
    title_list,abstract_list,y_list = load_data()
    stpwrdpath = "./model/stopwords.txt"
    stpwrd_dic = open(stpwrdpath, 'rb')
    stpwrd_content = stpwrd_dic.read()
    # 将停用词表转换为list
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()
    tag=["人事动态","公司信息","其他相关","成果奖项","财务经营","资本运作","风险危机"]
    label=[]
    for fb in y_list[0:400000]:
        if fb==tag[0]:
            label.append(0)
        elif fb ==tag[1]:
            label.append(1)
        elif fb == tag[2]:
            label.append(2)
        elif fb == tag[3]:
            label.append(3)
        elif fb == tag[4]:
            label.append(4)
        elif fb == tag[5]:
            label.append(5)
        elif fb == tag[6]:
            label.append(6)
        else:
            pass
    abstract_on=[]
    text=0
    for text in range(len(abstract_list[0:400000])):
        if abstract_list[0:400000][text]!=None and y_list[0:400000][text]!=None or abstract_list[0:400000][text]!=" " and y_list[0:400000][text]!=" ":
            abstract_on.append(abstract_list[0:400000][text])
        else:
            pass
        text=text+1
    numclass=len(list(set(label)))
    seq_list=[]
    seq_train=[]
    seq_context=[]
    seq_learn=[]
    seq_char=[]
    seq_context_char=[]
    a2=0
    for a2 in range(len(abstract_on)):
        if y_list[a2] !=None and abstract_on[a2]!=" " and abstract_on[a2]!=None :
            for b2 in abstract_on[a2]:
                if b2 not in stpwrdlst:
                    seq_char.append(b2)
            seq_context_char.append(str(seq_char))
            seq_char=[]
        else:
            pass
        a2=a2+1
    a1=0
    for a1 in range(len(abstract_on)):
        if y_list[a1] != None and abstract_on[a1]!=" " and abstract_on[a1]!=None:
            seq1=jieba.lcut(abstract_on[a1])
            for b1 in seq1:
                if b1 not in stpwrdlst:
                    seq_learn.append(b1)
                else:
                    pass
            seq_context.append(str(seq_learn))
            seq_learn=[]
        else:
            pass
        a1=a1+1
    tokenizer = Tokenizer(filters='!"#$%&()*+[]【,】-./:;<=>?@[\\]^_`{|}~\t')
    tokenizer.fit_on_texts(seq_context_char)
    wordindex = tokenizer.word_index
    wordindex['PAD'] = 0
    wordindex['UNK'] = 1
    # model=gensim.models.Word2Vec.load('./model/word2vec_wx')
    model = gensim.models.Word2Vec.load('./model/ner_daily.model')
    embedding_matrix = np.zeros((len(wordindex) + 1, 256))
    for word, i in wordindex.items():
        if word in model:
            embedding_matrix[i] = np.asarray(model[word])
        elif word not in model:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] =np.random.uniform(-0.25, 0.25, 256)
    len_list = []
    maxlen0 = 256
    X0 = tokenizer.texts_to_sequences(seq_train)
    X0 = pad_sequences(X0, maxlen=maxlen0)
    tokenizer.fit_on_texts(seq_context)
    wordindex1 = tokenizer.word_index
    wordindex1['PAD'] = 0
    wordindex1['UNK'] = 1
    # model=gensim.models.Word2Vec.load('./model/word2vec_wx')
    model1 = gensim.models.Word2Vec.load('./model/ner_daily.model')
    embedding_matrix1 = np.zeros((len(wordindex1) + 1, 256))
    for word1, i2 in wordindex1.items():
        if word1 in model1:
            embedding_matrix1[i2] = np.asarray(model1[word1])
        elif word1 not in model1:
            # words not found in embedding index will be all-zeros.
            embedding_matrix1[i2] = np.random.uniform(-0.25, 0.25, 256)
    X1 = tokenizer.texts_to_sequences(seq_context)
    maxlen1=maxlen0
    X1 = pad_sequences(X1, maxlen=maxlen1)
    X=X1
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(label)
    labels = np_utils.to_categorical(encoded_Y)
    matrix0=np.vstack((embedding_matrix,embedding_matrix1))#字词向量联合表示，于行上拼接
    y_pred_test=[]
    batch_size = 64
    n_epoch = 10
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.4, random_state=24)
    model=creat_model(wordindex, wordindex1, matrix0, maxlen0,X_train, X_test, y_train, y_test)
    y_predict = model.predict(X_test)
    y_pred = np.argmax(y_predict, axis=1)
    tag = ["人事动态", "公司信息", "其他相关", "成果奖项", "财务经营", "资本运作", "风险危机"]
    for count1 in list(y_pred):
        y_pred_test.append(tag[count1])
    print(y_pred_test)
    y_true = []
    for a in y_test:
        if list(a) == [1, 0, 0, 0, 0, 0, 0]:
            y_true.append(tag[0])
        if list(a) == [0, 1, 0, 0, 0, 0, 0]:
            y_true.append(tag[1])
        if list(a) == [0, 0, 1, 0, 0, 0, 0]:
            y_true.append(tag[2])
        if list(a) == [0, 0, 0, 1, 0, 0, 0]:
            y_true.append(tag[3])
        if list(a) == [0, 0, 0, 0, 1, 0, 0]:
            y_true.append(tag[4])
        if list(a) == [0, 0, 0, 0, 0, 1, 0]:
            y_true.append(tag[5])
        if list(a) == [0, 0, 0, 0, 0, 0, 1]:
            y_true.append(tag[6])
    target_names = tag
    print(classification_report(y_true, y_pred_test, target_names=target_names))
    print("check")
    print(1)
