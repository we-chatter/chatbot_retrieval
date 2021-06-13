"""
@author: 吴欣辉
@code: processors.py
@date: 2020-11-11
@desc: 预处理函数库
"""

import os, sys
import pickle
import tensorflow as tf
import numpy as np
from transformers import InputExample
from transformers import InputFeatures
from transformers import DataProcessor


class rankInputExample(InputExample):
    def __init__(self, guid, text1, text2, label):
        self.guid = guid
        self.text1 = text1
        self.text2 = text2
        self.label = label


class rankDataProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.txt"))

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.txt"))

    def _read_tsv(self, tsv):
        results = []
        fid = open(tsv, "r")
        for i, line in enumerate(fid.readlines()):
            if i == 0:
                continue
            line = line.strip().split("\t")
            results.append(tuple(line))
        fid.close()
        return results

    def _create_examples(self, tsv):
        examples = []
        srcs = self._read_tsv(tsv)
        for i, j in enumerate(srcs):
            examples.append(rankInputExample(
                guid=i,
                text1=j[1],
                text2=j[2],
                label=int(j[3])
            ))
        return examples


def rank_convert_examples_to_features(
        examples,
        max_seq_len,
        tokenizer,
        return_tensors=False
):
    features = []
    for j in examples:
        guid = j.guid
        text1 = j.text1
        text2 = j.text2
        label = j.label
        tokenizerids = tokenizer.encode_plus(
            text=text1,
            text_pair=text2,
            max_length=max_seq_len,
            pad_to_max_length=True,
            truncation=True
        )

        input_ids = tokenizerids["input_ids"]
        token_type_ids = tokenizerids["token_type_ids"]
        attention_mask = tokenizerids["attention_mask"]

        assert len(input_ids) == max_seq_len
        assert len(token_type_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len

        features.append(InputFeatures(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            label=label
        ))
    if return_tensors:
        def generater():
            for j in features:
                yield (
                    {
                        "input_ids": j.input_ids,
                        "token_type_ids": j.token_type_ids,
                        "attention_mask": j.attention_mask
                    },
                    j.label
                )

        datasets = tf.data.Dataset.from_generator(
            generater,
            (
                {
                    "input_ids": tf.int32,
                    "token_type_ids": tf.int32,
                    "attention_mask": tf.int32
                },
                tf.int32,
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None])
                },
                tf.TensorShape([]),
            )
        )
        return datasets