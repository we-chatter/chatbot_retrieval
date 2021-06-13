import json
import sys

# sys.path.append("../")
# from model.rankONNX import rankONNX
from xiaodingdang.services.rank.models.rankONNX import rankONNX

Config = json.load(open("../saved_models/rank_bert_20201111/rank_setting.json", "r"))
models = rankONNX(Config)
logits = models.predict([("人才补贴", "余杭高层次人才租房补贴")])
print(logits)
