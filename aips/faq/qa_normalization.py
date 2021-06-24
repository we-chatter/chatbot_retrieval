# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   qa_normalization.py
 
@Time    :   2020/10/3 12:15 下午
 
@Desc    :   检索数据后处理，步骤：
             1、基于es倒排索引的粗排序
             2、bert + onnx的语义相似度匹配精排
 
"""
from models.QA.es_qa import Searcher

from models.rank.rank import rank_instance

es = Searcher()


def qa_normal(message):
    """
    问答标准化
    """

    response = es.search_es(message)
    responses_sorted = sorted(response, key=lambda x: x['score'], reverse=True)    # 倒排索引粗排序后召回结果
    result = rank_instance.do_rank(message, responses_sorted)

    return result
