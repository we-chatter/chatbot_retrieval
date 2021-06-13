# -*- coding: utf-8 -*-

"""
@Author  :   Xu

@Software:   PyCharm

@File    :   train.py

@Time    :   2019-06-10 14:44

@Desc    :  模型训练

"""

import json
import os
import sys
import pathlib

from models.rank.model.rankBert import rankBert

basedir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if __name__ == "__main__":
    config = json.load(open("../../saved_models/rank_bert_20201111/rank_setting.json", "r"))
    srcs = basedir + '/data'
    models = rankBert(config, 1)
    models.train(srcs)
    models.test(srcs)
    logits = models.predict([("天气真不错", "好棒的天气")])
    print(logits)
