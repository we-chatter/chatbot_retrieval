# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   MysqlUtils.py
 
@Time    :   2020/6/12 3:13 下午
 
@Desc    :   数据查询
             用到的表（离线表）：
"""

import pymysql
import logging


logger = logging.getLogger(__name__)


class FAQData:

    def __init__(self):

        # 本地
        self.database = 'chatbot_cn'
        self.host = 'localhost'
        self.username = '111'
        self.password = '111'
        self.db = pymysql.connect(host=self.host, port=3306, user=self.username, passwd=self.password, db=self.database)

    def get_faq_data(self):
        logger.info('Connect mysql {} successfully'.format(self.db))
        cursor = self.db.cursor()
        cursor.execute("select id,"
                       "question,"
                       "answer,"
                       "category"
                       " from Chatbot_Retrieval_model_faq ")
        res = cursor.fetchall()
        logger.info('All resource length is {}'.format(len(res)))
        # cursor.close()
        # self.db.close()

        return res


if __name__ == "__main__":
    speechId = '569353302990019129'
    FAQData().get_faq_data()