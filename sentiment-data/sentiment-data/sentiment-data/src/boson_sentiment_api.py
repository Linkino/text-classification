# -*- coding: utf-8 -*-
# ######################################################################################################################
# 文件名：boson_sentiment_api.py
# 描述：读取文本文件
# 作者：windy
# 时间：2018.03.24
# ######################################################################################################################

import re
import os
import json
import time
import requests
from src.common.datasql import *
from src.common.datatxt import *

pydir  = os.path.split(os.path.realpath(__file__))[0]
dictdir = os.path.abspath(os.path.join(pydir,os.path.pardir))
dbpath = os.path.join(dictdir,"./config/mysql.con")
datasql = DataSQL(dbpath= dbpath)

# post_json数据
def post_json(url,jsondata):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data = jsondata, headers = headers)
        # response.raise_for_status()
        return response
    except TimeoutError:
        print("=======================================调用接口异常====================================================")
        return None

def boson_sentiment_api(text):
    url='https://bosonnlp.com/analysis/sentiment'
    data = {'analysisType': 'news','data':text}
    jsondata = json.dumps(data)
    resp = post_json(url,jsondata)
    result = json.loads(resp.text)
    score = result[0][0]
    return score

# 情感分析接口
def sentiment_api(sentence,token):
    SENTIMENT_URL = 'http://api.bosonnlp.com/sentiment/analysis'
    headers = {'X-Token': token}
    data = json.dumps(sentence)
    try:
        resp = requests.post(SENTIMENT_URL, headers=headers, data=data.encode('utf-8'))
        score = resp.text.replace("[",'').replace("]",'').split(',')[0]
        time.sleep(2)
    except:
        print("=======================================调用接口异常====================================================")
        score = ""
    return score


def write_sql(data):
    datasql = DataSQL(dbpath= dbpath)
    sql = "insert into data_label_boson (nid,score) values (%s,%s)"
    n = datasql.insert_data(data,sql)

def main():
    data_list = datasql.load_table(sql = "select id,title,abstract from data_label where id >10000  limit 10000" )
    for k in range(len(data_list)):
        data = data_list[k]
        nid = data[0]
        text = data[1]+'。\t'+data[2]
        try:
            print(str(nid)+".================================================================================")
            score = boson_sentiment_api(text)
            data = (nid,score)
            write_sql(data)
            print(score)
        except:
            print(str(nid)+".================================================================================")
            print("error")
        time.sleep(3)

def load_tokens(path):
    tokens = read_file(path)
    return tokens

def main2():
    token = "aYZGwJzu.19202.Fa000JVzIkpR"
    tokens = load_tokens('../config/token.txt')
    data_list = datasql.load_table(sql = "select id,title,abstract from data_label where id >10000  limit 10000" )
    token = tokens.pop()
    for k in range(len(data_list)):
        data = data_list[k]
        nid = data[0]
        text = data[1]+'。\t'+data[2]
        try:
            print(str(nid)+".================================================================================")
            score = sentiment_api(text,token)
            data = (nid,score)
            write_sql(data)
            print(score)
            if k % 500 == 0:
                token = tokens.pop()
                print(token)
        except:
            print(str(nid)+".================================================================================")
            print("error")
        time.sleep(3)



if __name__ == "__main__":
    text = "大数据造日版机器邓亚萍！打法没法模仿，国乒绝招失灵，刘国梁麻烦大了！"
    # resp = boson_sentiment_api(json.dumps(text))
    # main()
    # token = "sRV3r8Zg.28046.QsEdhR4UJH8U"
    # score = sentiment_api(text,token)
    # print(score)

    main()