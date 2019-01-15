# -*- coding: utf-8 -*-
# ######################################################################################################################
# 说明：获取数据
# 文件：TextData.py
# 时间：2017.11.07
# 作者：wjian
# ######################################################################################################################

import pymysql

class TextData:
    def __init__(self):
        self.sql =  'SELECT * from icekredit_nlp_data'
        self.titleList = []
        self.contextList = []
        self.labelList = []
        self.newsidList = []

    # ======================================================================================================================
    # 读取配置文件
    # ======================================================================================================================
    def loadConfig(self,openPath):
        file = open(openPath,'r')
        textlines = file.readlines() # 读取全部内容
        host =   textlines[1].split(":")[1].replace("\n","")
        port =   textlines[2].split(":")[1].replace("\n","")
        dbname = textlines[3].split(":")[1].replace("\n","")
        user =   textlines[4].split(":")[1].replace("\n","")
        pwd  =   textlines[5].split(":")[1].replace("\n","")
        file.close()
        return host,port,dbname,user,pwd

    # ======================================================================================================================
    # 连接数据库
    # ======================================================================================================================
    def connectdb(self,path='./config/mysql_themedata.con'):
        host,port,dbname,user,pwd = self.loadConfig(path)
        db = pymysql.connect(host=host,port = int(port),user=user,  passwd=pwd,db =dbname,charset ='utf8')
        return db

    # ======================================================================================================================
    # 获取数据表中所有数据
    # ======================================================================================================================
    def load_table(self,sql,dbpath='./config/mysql_themedata.con'):
        db = self.connectdb(path=dbpath)
        with db:
            cursor = db.cursor()
            cursor.execute(sql)
            data_rows = cursor.fetchall()
        cursor.close()
        db.commit()
        db.close()
        return data_rows

if __name__ == "__main__":

    pass









