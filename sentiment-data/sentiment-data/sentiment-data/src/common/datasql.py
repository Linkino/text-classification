# -*- coding: utf-8 -*-
# ######################################################################################################################
# 文件名：datasql.py
# 描述：读取文本文件
# 作者：windy
# 时间：2018.03.24
# ######################################################################################################################

import os
import pymysql

pydir  = os.path.split(os.path.realpath(__file__))[0]
dictdir = os.path.abspath(os.path.join(pydir,os.path.pardir))
dbpath = os.path.join(dictdir,"./config/mysql.con")
print(dbpath)


class DataSQL(object):
    def __init__(self,dbpath):
        self.dbpath = dbpath

    # ==================================================================================================================
    # 读取配置文件
    # ==================================================================================================================
    def loadConfig(self):
        file = open(self.dbpath,'r')
        textlines = file.readlines() # 读取全部内容
        host =   textlines[1].split(":")[1].replace("\n","")
        port =   textlines[2].split(":")[1].replace("\n","")
        dbname = textlines[3].split(":")[1].replace("\n","")
        user =  textlines[4].split(":")[1].replace("\n","")
        pwd  =  textlines[5].split(":")[1].replace("\n","")
        file.close()
        return host,port,dbname,user,pwd

    # ==================================================================================================================
    # 连接数据库
    # ==================================================================================================================
    def connectdb(self):
        host,port,dbname,user,pwd = self.loadConfig()
        db = pymysql.connect(host=host,port = int(port),user=user,  passwd=pwd,db =dbname,charset ='utf8mb4')
        return db

    # ==================================================================================================================
    # 获取数据表中所有数据
    # ==================================================================================================================
    def load_table(self,sql):
        db = self.connectdb()
        data_rows = []
        try:
            with db:
                cursor = db.cursor()
                cursor.execute(sql)
                data_rows = cursor.fetchall()
            cursor.close()
            db.commit()
            db.close()
        except:
            print("加载数据异常")
        return data_rows

    # ======================================================================================================================
    # 批量存储数据集
    # ======================================================================================================================
    def insert_data_list(self,sql,data_list,batch_size = 5):
        batch_list = []             # 每一批插入的数据量
        batch_index = 0             # 每一批的索引
        import_count = 0            # 导入的数据量
        left_count = len(data_list) # 还未导入的数据量
        # 连接数据库
        db = self.connectdb()
        cursor = db.cursor()
        # SQL 插入语句
        # 批量插入
        for k in range(len(data_list)):
            data = data_list[k]  # 数据为元组格式
            batch_list.append(data)
            batch_index = batch_index + 1
            left_count = left_count - 1
            if batch_index == batch_size or left_count == 0:
                try:
                # 执行sql语句
                    cursor.executemany(sql, batch_list)
                    import_count = import_count + cursor.rowcount
                    db.commit()
                    batch_list = []
                    batch_index = 0
                    print("存入数据:%s"%import_count)
                except:
                    # 发生错误时回滚
                    db.rollback()
                    batch_list = []
                    batch_index = 0
                    print("error:存入数据")
        # 关闭数据库连接
        db.close()
        return import_count

    # ======================================================================================================================
    # 存储数据集
    # ======================================================================================================================
    def insert_data(self,data,sql):
        # 连接数据库
        db = self.connectdb()
        cursor = db.cursor()
        # SQL 插入语句
        # sql = "INSERT INTO data_sentence (nid, sentence, ner,title) VALUES (%s, %s, %s, %s)"
        try:
            cursor.execute(sql,data)
            db.commit()
        except:
            db.rollback() # 发生错误时回滚
        db.close() # 关闭数据库连接


if __name__ == "__main__":
    sql = "select m.id,m.title,m.url_name from data_source m order by createdtm desc limit 5000"
    dataSQL = DataSQL(dbpath)
    data_rows = dataSQL.load_table(sql)
    pass

