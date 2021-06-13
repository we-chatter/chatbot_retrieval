# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   preprocess.py
 
@Time    :   2020/12/10 10:15 上午
 
@Desc    :
 
"""
# import pandas as pd


def read_src_data(filein):
    """

    """
    with open(file, 'r', encoding='utf-8') as fin:
        row = fin.readlines()
        for one in row:
            print(one)


if __name__=='__main__':
    file = '/Data/xiaobensuan/Codes/chatbot_retrieval/data/train.txt'
    read_src_data(file)