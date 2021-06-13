# -*- coding: utf-8 -*-

"""
@Author  :   Xu

@Software:   PyCharm

@File    :   rankBert.py

@Time    :   2019-06-10 14:44

@Desc    :

"""

import os

import keras2onnx
import tensorflow as tf
from onnxruntime_tools import optimizer
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from models.rank.utils.processors import rankDataProcessor
from models.rank.utils.processors import rank_convert_examples_to_features


def tf_keras_convert_to_onnx(models, paths, config):
    onnxRankBert = keras2onnx.convert_keras(
        models,
        models.name,
        target_opset=12
    )
    keras2onnx.save_model(
        onnxRankBert,
        paths
    )
    optimized_model = optimizer.optimize_model(
        paths,
        model_type='bert_keras',
        num_heads=config.num_attention_heads,
        hidden_size=config.hidden_size
    )
    optimized_model.use_dynamic_axes()
    optimized_model.save_model_to_file(
        paths
    )


def calculate_steps(examples_length, batch_size):
    if examples_length % batch_size == 0:
        tmp_steps = examples_length // batch_size
    else:
        tmp_steps = examples_length // batch_size + 1
    return tmp_steps


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


class rankBert(object):
    def __init__(self, args, type_ids):
        super(rankBert, self).__init__()
        self.max_seq_len = args["max_seq_len"]
        self.batch_size = args["batch_size"]
        self.epochs = args["epochs"]
        self.model_path = args["models_path"]
        self.pretrain_model_path = args["pretrain_model_path"]
        self.rankDataProcessor = rankDataProcessor()
        self.rankTokenizer = rankTokenizer.from_pretrained(
            os.path.join(self.pretrain_model_path, "vocab.txt"),
            do_lower_case=True
        )
        self.__model_builder__(type_ids)
        self.__model_compile__()

    def __model_builder__(self, type_ids):
        if type_ids:
            self.bertConfig = BertConfig.from_pretrained(
                os.path.join(self.pretrain_model_path, "config.json"),
                num_labels=2
            )
            self.rankBert = TFBertForSequenceClassification.from_pretrained(
                os.path.join(self.pretrain_model_path, "tf_model.h5"),
                config=self.bertConfig
            )
        else:
            self.bertConfig = BertConfig.from_pretrained(
                os.path.join(self.model_path, "config.json"),
            )
            self.rankBert = TFBertForSequenceClassification.from_pretrained(
                os.path.join(self.model_path, "tf_model.h5"),
                config=self.bertConfig
            )

    def __model_compile__(self):
        self.losses = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizers = tf.keras.optimizers.Adam(
            learning_rate=1e-5
        )
        self.metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    def train(self, srcs):
        train_examples = self.rankDataProcessor.get_train_examples(srcs)
        dev_examples = self.rankDataProcessor.get_dev_examples(srcs)
        train_datasets = rank_convert_examples_to_features(
            train_examples,
            self.max_seq_len,
            self.rankTokenizer,
            return_tensors="tf"
        )
        dev_datasets = rank_convert_examples_to_features(
            dev_examples,
            self.max_seq_len,
            self.rankTokenizer,
            return_tensors="tf"
        )
        train_datasets = train_datasets.shuffle(10000)
        train_datasets = train_datasets.batch(self.batch_size)
        dev_datasets = dev_datasets.batch(self.batch_size)

        best_val_acc = 0
        train_steps = calculate_steps(len(train_examples), self.batch_size)
        dev_steps = calculate_steps(len(dev_examples), self.batch_size)
        print("<==================================={}===================================>".format("training"))
        for j in range(self.epochs):
            print("<==================================={}===================================>".format(
                "epochs ID>> " + str(j)))
            train_bar = tf.keras.utils.Progbar(train_steps, stateful_metrics=['loss', 'acc'])
            for train_step, (x, y) in enumerate(train_datasets):
                with tf.GradientTape() as tape:
                    logits = self.rankBert(x, training=True)[0]
                    cross_entropy_loss = tf.reduce_sum(self.losses(y, logits)) * (1.0 / self.batch_size)
                    self.metrics.update_state(y, logits)
                    accuracy = self.metrics.result()
                grads = tape.gradient(cross_entropy_loss, self.rankBert.trainable_weights)
                self.optimizers.apply_gradients(zip(grads, self.rankBert.trainable_weights))
                train_bar.update(train_step + 1, values=[("loss", float(cross_entropy_loss)), ("acc", float(accuracy))])

            self.metrics.reset_states()
            dev_bar = tf.keras.utils.Progbar(dev_steps, stateful_metrics=['val_acc'])
            for dev_step, (x, y) in enumerate(dev_datasets):
                logits = self.rankBert(x, training=False)[0]
                self.metrics.update_state(y, logits)
                accuracy = self.metrics.result()
                dev_bar.update(dev_step + 1, values=[("val_acc", float(accuracy))])

            val_acc = self.metrics.result()
            if val_acc > best_val_acc:
                self.rankBert.save_pretrained(self.model_path)
                best_val_acc = val_acc

            self.metrics.reset_states()
            print("")

        temps = [("天气真不错", "天气真好")]
        temp_features = self.rankTokenizer.batch_encode_plus(
            batch_text_or_text_pairs=temps,
            return_tensors="tf",
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            truncation=True
        )
        self.rankBert.predict(temp_features.data)

        tf_keras_convert_to_onnx(
            self.rankBert,
            os.path.join(self.model_path, "rankBert.onnx"),
            self.bertConfig
        )

    def test(self, srcs):
        test_examples = self.rankDataProcessor.get_test_examples(srcs)
        test_datasets = rank_convert_examples_to_features(
            test_examples,
            self.max_seq_len,
            self.rankTokenizer,
            return_tensors="tf"
        )
        test_datasets = test_datasets.batch(self.batch_size)
        test_steps = calculate_steps(len(test_examples), self.batch_size)
        test_bar = tf.keras.utils.Progbar(test_steps, stateful_metrics=['test_acc'])
        print("<==================================={}===================================>".format("testing"))
        for test_step, (x, y) in enumerate(test_datasets):
            logits = self.rankBert(x, training=False)[0]
            self.metrics.update_state(y, logits)
            accuracy = self.metrics.result()
            test_bar.update(test_step + 1, values=[("test_acc", float(accuracy))])
        self.metrics.reset_states()

    def predict(self, sentences):
        features = self.rankTokenizer.batch_encode_plus(
            sentences,
            return_tensors="tf",
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            truncation=True
        )
        logits = self.rankBert(features, training=False)[0]
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        similarity = [j[1] for j in probs]
        return similarity


if __name__ == "__main__":
    text = [("天气真不错", "天气真好")]
    rankTokenizer = rankTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext/vocab.txt")
    print(rankTokenizer.tokenize(text))
    print(rankTokenizer.encode_plus(text))
