#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import numpy as np
import tensorflow as tf
from nlpgnn.models import TextCNN
from nlpgnn.optimizers import optim
from nlpgnn.metrics import Metric, Losess
from nlpgnn.datas.dataloader import TFWriter, TFLoader

maxlen = 128
batch_size = 64
embedding_dims = 100
vocab_file = "Input/vocab.txt"
class_num = 2
vocab_size = 30522  # line in vocab.txt

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(maxlen,
                    vocab_file,
                    modes=["valid"],
                    task='cls',
                    check_exist=False)

load = TFLoader(maxlen,
                  batch_size,
                  task='cls',
                  epoch=1)

model = TextCNN.TextCNN(maxlen, vocab_size, embedding_dims, class_num)

f1score = Metric.SparseF1Score(average="macro")
precsionscore = Metric.SparsePrecisionScore(average="macro")
recallscore = Metric.SparseRecallScore(average="macro")
accuarcyscore = Metric.SparseAccuracy()

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('./save'))
# For test model
Batch = 0
f1s = []
precisions = []
recalls = []
accuracys = []
for X, token_type_id, input_mask, Y in load.load_valid():
    with tf.GradientTape() as tape:
        predict = model.predict(X)
        f1s.append(f1score(Y, predict))
        precisions.append(precsionscore(Y, predict))
        recalls.append(recallscore(Y, predict))
        accuracys.append(accuarcyscore(Y, predict))
        # print("Sentence", writer.convert_id_to_vocab(tf.reshape(X, [-1]).numpy()))
        #
        # print("Label", writer.convert_id_to_label(tf.reshape(predict, [-1]).numpy()))
print("f1:{}\tprecision:{}\trecall:{}\taccuracy:{}\n".format(np.mean(f1s),
                                                             np.mean(precisions),
                                                             np.mean(recalls),
                                                             np.mean(accuracys)))
