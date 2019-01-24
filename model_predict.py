import flask
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from attention import AttentionWithContext
from keras.models import load_model
import jieba
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from data_db import TextData
import pickle
import json

def load_data():
    sql = "select context,abstract,id from theme_data_test"
    pTextData = TextData()
    data_rows = pTextData.load_table(sql)
    title_list = []
    abstract_list = []
    for row in data_rows:
        title = row[0]
        content = row[1]
        title_list.append(title)
        abstract_list.append(content)
    return title_list,abstract_list
data=[]
title_list,abstract_list = load_data()
stpwrdpath = "./model/stopwords.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
# 将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()
a1=0
seq_learn = []
seq_context=[]
title_list=title_list[0:]
for a1 in range(len(title_list)):
    if type(title_list[a1]) != None:
        seq1 = jieba.lcut(title_list[a1])
        for b1 in seq1:
            if b1 not in stpwrdlst:
                seq_learn.append(b1)
            else:
                pass
        seq_context.append(seq_learn)
        seq_learn = []
    else:
        pass
    a1 = a1 + 1
pkl_file = open('./dic/data_2.pkl', 'rb')
data1 = pickle.load(pkl_file)
pkl_file.close()
w=0
e=0
for w in range(len(seq_context)):
    for e in range(len(seq_context[w])):
        if str(seq_context[w][e]) not in data1:
            seq_context[w][e]=int(1)
        elif str(seq_context[w][e]) in data1:
            seq_context[w][e]=int(data1[str(seq_context[w][e])])
        e=e+1
    w=w+1
seq_context=pad_sequences(seq_context, maxlen=256)
filepath = "./model/sen_model_title_textcnn_yh_1.h5"
model = load_model(filepath, compile=False, custom_objects={'AttentionWithContext': AttentionWithContext})
pre = model.predict(seq_context)
y_pred_test=[]
y_pred = np.argmax(pre, axis=1)
tag = ["neg", "neu", "pos"]
for count in list(y_pred):
    y_pred_test.append(tag[count])
y_pred_test = list(y_pred_test)
print("____________________________________________预测结果_____________________________________")
k=0

f2 ="./data/predict_test_context_data.txt"
with open(f2,"w",encoding="utf-8") as file:
    for k in range(len(seq_context)):
        if str(title_list[k]) !=" " and type(str(title_list[k]))!=None and str(title_list[k])!=None:
            print("数据项：",str(title_list[k]))
            file.write(str(k+1)+"."+"数据项："+str(title_list[k])+ "\n")
            print("对应标签:",y_pred_test[k])
            file.write("对应标签：" + str(y_pred_test[k]) + "\n")
        else:
            pass
        k=k+1
file.close()
print(1)