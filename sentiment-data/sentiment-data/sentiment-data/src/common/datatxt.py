# -*- coding: utf-8 -*-
# ######################################################################################################################
# 文件名：datatxt.py
# 描述：读取文本文件
# 作者：windy
# 时间：2018.03.24
# ######################################################################################################################

import csv
import pandas as pd

# ==================================================================================================================
# 读取文本文件
# ==================================================================================================================
def read_file(path):
    try:
        with open(path,encoding='utf-8') as f: #
            lines = f.readlines()
    except IOError:
        with open(path) as f:
            lines = f.readlines()
    data = []
    for line in lines:
        data.append(line.strip('\n').strip())
    return data

# ==================================================================================================================
# 保存文本文件
# ==================================================================================================================
def write_file(path,data_list):
    try:
        pfile = open(path,'w',encoding='utf-8')
        for k in range(len(data_list)):
            row = data_list[k]+"\n"
            pfile.write(row)
        pfile.close()
    except IOError:
        pass

# ==================================================================================================================
# 保存文本文件
# ==================================================================================================================
def write_file_dict(path,data_dict):
    try:
        pfile = open(path,'w',encoding='utf-8')
        for k,v in data_dict.items():
            row = k+"\t"+v+"\n"
            pfile.write(row)
        pfile.close()
    except IOError:
        pass

# ==================================================================================================================
# 读取文本文件
# ==================================================================================================================
def read_file_dict(path):
    try:
        with open(path,encoding='utf-8') as f: #
            lines = f.readlines()
    except IOError:
        with open(path) as f:
            lines = f.readlines()
    data = {}
    for line in lines:
        k,v = line.strip('\n').split('\t')
        data[k] = v
    return data

# ==================================================================================================================
# 读取csv文件
# ==================================================================================================================
def read_csv(path):
    try:
        csv_file = open(path, encoding='utf-8')
        csv_data = csv.reader(csv_file)
        csv_file.close()
    except IOError:
        csv_data = []
    return csv_data

# ==================================================================================================================
# 保存csv文件
# ==================================================================================================================
def write_csv(path,data_list):
    try:
        pfile = open(path,'w',encoding='utf-8',newline='')
        csv_writer = csv.writer(pfile, dialect='excel')
        csv_writer.writerow(data_list)
        pfile.close()
    except IOError:
        pass

# ==================================================================================================================
# pandas读取csv文件
# ==================================================================================================================
def read_csv_pd(path,sep = '\t'):
    try:
        csv_data = pd.read_csv(path,encoding = 'utf-8',sep = sep)
    except IOError:
        csv_data = []
    return csv_data



