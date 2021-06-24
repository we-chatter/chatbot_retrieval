# -*- coding: utf-8 -*-

"""
@Author  :   Xu

@Software:   PyCharm

@File    :   run_server.py

@Time    :   2020/8/26 2:50 下午

@Desc    :

"""
import datetime
import json
import os
import sys
import logging

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from config import CONFIG

from sanic import Sanic, response
from sanic.response import text, HTTPResponse
from sanic.request import Request
from utils.LogUtils import Logger

from aips.faq.qa_normalization import qa_normal
from models.rank.rank import rank_instance

logger = logging.getLogger(__name__)

app = Sanic(__name__)

app.update_config(CONFIG)


@app.route("/")
async def test(request):
    return text('Welcome to the faq platform')


@app.post('/faq')
async def faq_predict(request: Request) -> HTTPResponse:
    """

    :return:
    """
    localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        sender = request.json["sender"]
        querys = request.json["message"]
        if "fields" in request.json:
            fields = request.json["fields"]
        if "score" in request.json:
            score = request.json["score"]
        result = qa_normal(querys)
        res_dic = {
            "status": "success",
            "code": 200,
            "result": result,
            "time": localtime
        }
        log_res = json.dumps(res_dic, ensure_ascii=False)
        logger.info(log_res)
        return response.json(res_dic,
                             dumps=json.dumps)
    except Exception as e:
        logger.info('Error is {}'.format(e))
        res_dic = {
            "status": 'Failure',
            "code": 400,
            "result": str(e),
            "time": localtime
        }

        return response.json(res_dic)


@app.post('/match')
async def sematic_match(request: Request) -> HTTPResponse:
    """
    语义匹配
    """
    try:
        querys = request.json["message"]
        result = rank_instance.do_match(querys)
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res_dic = {
            "result": float(result[0]),
            "time": localtime
        }
        log_res = json.dumps(res_dic, ensure_ascii=False)
        logger.info(log_res)
        return response.json(res_dic,
                             dumps=json.dumps)
    except Exception as e:
        logger.info(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9020, auto_reload=True, workers=1)