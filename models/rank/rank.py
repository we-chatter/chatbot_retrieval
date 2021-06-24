# -*- coding: utf-8 -*-

"""
@Author  :   Xu

@Software:   PyCharm

@File    :   LogUtils2.py

@Time    :   2018-06-13 17:51

@Desc    :

"""

import logging
import string

from models.rank.model.rankONNX import RankONNX


logger = logging.getLogger(__name__)


class RankV1(object):
    """docstring for rankAPi"""

    def __init__(self):
        logging.info("===========================排序接口开始加载===========================")
        # models_path = "rank/saved_models/rank_bert_20201111"
        # dir_model = os.path.join(settings.RESOURCES_DIR, "rank/saved_models/rank_bert_20201111")
        # path_json = os.path.join(dir_model, "rank_setting.json")
        # logging.info(f"==> 加载排序接口的模型配置文件: {path_json}")
        # config = json.load(open(path_json))
        self.rankONNX = RankONNX()
        logging.info("===========================排序接口加载完成===========================")

    def do_match(self, json_data):
        logging.info(f"执行rank排序...")
        # l_recall_results = json_data["results"]
        # question = json_data["question"].strip(string.punctuation)
        # querys = list(zip([question] * len(l_recall_results), [j["question"].strip(string.punctuation) for j in l_recall_results]))
        pre_data = []
        sss = tuple(json_data)
        pre_data.append(sss)
        rank_scores = self.rankONNX.predict(pre_data)
        # for j in range(len(l_recall_results)):
        #     l_recall_results[j]["rank_score"] = float(rank_scores[j]) * 0.3 + float(l_recall_results[j]["recall_score"])
        # max_scores = max([j["rank_score"] for j in l_recall_results])
        # recalls = [j for j in l_recall_results if j["rank_score"] >= max_scores - 0.25]
        # recalls = [j for j in recalls if j["rank_score"] >= 0.15]
        # logger.info(f"rank结果集: {recalls}")
        # if recalls:
        #     recalls = sorted(recalls, key=lambda r: r["rank_score"], reverse=True)
        #     json_data["results"] = recalls
        #     if len(recalls) == 1:
        #         json_data["has_hits"] = 2
        return rank_scores

    def do_rank(self, query, esres):
        """
        @params: query 用户说的话
        @params: esres es粗排召回结果
        """
        logging.info(f"执行rank排序...")
        querys = list(zip([query] * len(esres), [j["sim_question"].strip(string.punctuation) for j in esres]))
        logging.info(f"粗排序的结果是： {querys}")
        if querys:
            rank_scores = self.rankONNX.predict(querys)
            for j in range(len(esres)):
                esres[j]["score"] = float(rank_scores[j]) * 0.3 + float(esres[j]["score"])
            max_scores = max([j["score"] for j in esres])
            recalls = [j for j in esres if j["score"] >= max_scores - 0.25]
            recalls = [j for j in recalls if j["score"] >= 0.15]
            logger.info(f"rank结果集: {recalls}")
            # 这里要再排序一次
            # if recalls:
            #     recalls = sorted(recalls, key=lambda r: r["score"], reverse=True)
            return recalls
        else:
            # 这里是没有匹配结果的情况
            recalls = {
                "score": 0,
                "matches": 0,
                "sim_question": "",
                "answers": ["知识库没有查询到对应的数据"]
            }
            return recalls


rank_instance = RankV1()


if __name__ == "__main__":
    # RankV1().app("jsonfile")
    pass
