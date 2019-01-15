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
import json
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# app.run(host='127.0.0.1', port='4000')
global graph
graph = tf.get_default_graph()


@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}
    params = flask.request.json
    if (params == None):
        params = flask.request.args
    if (params != None):
        data["context"] = params.get("context")
        data["title"]= params.get("title")
        print(data)
        tokenizer = Tokenizer(filters='!"#$%&()*+[]【,】-./:;<=>?@[\\]^_`{|}~\t')
        x1 = [str(jieba.lcut(str(data['context'])))]
        print(x1)
        tokenizer.fit_on_texts(x1)
        wordindex = tokenizer.word_index
        wordindex['PAD'] = 0
        wordindex['UNK'] = 1
        print(wordindex)
        x = tokenizer.texts_to_sequences(x1)
        x = pad_sequences(x, maxlen=256)
        print(x)
        with graph.as_default():
            filepath = "./model/sen_model10_0.96.h5"
            model = load_model(filepath, compile=False, custom_objects={'AttentionWithContext': AttentionWithContext})
            pre = model.predict(x)
            y_pred_test=[]
            y_pred = np.argmax(pre, axis=1)
            tag = ["neg", "neu", "pos"]
            for count in list(y_pred):
                y_pred_test.append(tag[count])
            y_pred_test = list(y_pred_test)
            data["prediction"] = str(pre[0])
            data["label"] = y_pred_test[0]
            data["success"] = True
            print(pre)
    strdata=flask.jsonify(data)
    print(strdata)
    return strdata
app.run(host='0.0.0.0')
