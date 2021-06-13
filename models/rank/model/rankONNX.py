"""
@author: 吴欣辉
@code: rankONNX.py
@date: 2020-11-11
@desc: 排序ONNX模型
"""

import os
import onnxruntime
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'


def batch_generater(features, batch_size=128):
    input_ids = features["input_ids"].numpy()
    token_type_ids = features["token_type_ids"].numpy()
    attention_mask = features["attention_mask"].numpy()

    if len(input_ids) % batch_size == 0:
        n = len(input_ids) // batch_size
    else:
        n = len(input_ids) // batch_size + 1

    for j in range(n):
        yield {
            "input_ids": input_ids[j * batch_size:(j + 1) * batch_size],
            "token_type_ids": token_type_ids[j * batch_size:(j + 1) * batch_size],
            "attention_mask": attention_mask[j * batch_size:(j + 1) * batch_size]
        }


class rankTokenizer(BertTokenizer):
    def _tokenize(self, text):
        tokens = []
        for curPiece in text:
            if curPiece in self.vocab:
                tokens.append(curPiece)
            elif curPiece.isspace():
                tokens.append("[unused1]")
            else:
                tokens.append("[UNK]")
        return tokens


class rankONNX(object):
    def __init__(self, config):
        super(rankONNX, self).__init__()
        self.model_path = config["models_path"]
        self.max_seq_len = config["max_seq_len"]
        self.pretrain_model_path = config["pretrain_model_path"]
        self.rankTokenizer = rankTokenizer.from_pretrained(
            os.path.join(self.pretrain_model_path, "vocab.txt"),
            do_lower_case=True
        )
        self.sess_options = onnxruntime.SessionOptions()
        self.rankBertONNX = onnxruntime.InferenceSession(
            os.path.join(self.model_path, "rankBert.onnx"),
            self.sess_options,
            providers=["CUDAExecutionProvider"]
        )

    def predict(self, sentences):
        features = self.rankTokenizer.batch_encode_plus(
            sentences,
            return_tensors="tf",
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            truncation=True
        )
        batchs = batch_generater(features)
        logits = np.vstack([self.rankBertONNX.run(None, b)[0] for b in batchs])
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        similarity = [j[1] for j in probs]
        return similarity


class RankONNX(object):
    def __init__(self, ):
        self.model_path = '/data/xiaobensuan/Codes/chatbot_retrieval/models/saved_models/rank_bert_20201111'
        self.max_seq_len = 64
        self.pretrain_model_path = '/data/pretrain/tensorflow2.x/chinese-roberta-wwm-ext-large'
        self.rankTokenizer = rankTokenizer.from_pretrained(
            os.path.join(self.pretrain_model_path, "vocab.txt"),
            do_lower_case=True
        )
        self.sess_options = onnxruntime.SessionOptions()
        self.rankBertONNX = onnxruntime.InferenceSession(
            os.path.join(self.model_path, "rankBert.onnx"),
            self.sess_options,
            providers=["CUDAExecutionProvider"]
        )

    def predict(self, sentences):
        features = self.rankTokenizer.batch_encode_plus(
            sentences,
            return_tensors="tf",
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            truncation=True
        )
        batchs = batch_generater(features)
        logits = np.vstack([self.rankBertONNX.run(None, b)[0] for b in batchs])
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        similarity = [j[1] for j in probs]
        return similarity


if __name__ == "__main__":
    pass
