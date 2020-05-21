#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf
import numpy as np
from nlpgnn.models import TextCNN
from nlpgnn.optimizers import optim
from nlpgnn.metrics import Metric, Losess
from nlpgnn.datas.dataloader import TFWriter, TFLoader

maxlen = 128
batch_size = 64
embedding_dims = 100
vocab_file = "Input/vocab.txt"
word2vec = "./corpus/word2vec.vector"
class_num = 2
vocab_size = 30522  # line in vocab.txt

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(maxlen, vocab_file, modes=["train"], task='cls', check_exist=False)

load = TFLoader(maxlen, batch_size, task='cls', epoch=3)

# init_weights = writer.get_init_weight(word2vec,
#                                       vocab_size,
#                                       embedding_dims)

model = TextCNN.TextCNN(maxlen,
                        vocab_size,
                        embedding_dims,
                        class_num,
                        # init_weights,
                        weights_trainable=True)

# model = TextCNN.TextCNN(maxlen, vocab_size, embedding_dims, class_num)

# 构建优化器
# lr = tf.keras.optimizers.schedules.PolynomialDecay(0.01, decay_steps=18000,
#                                                    end_learning_rate=0.0001,
#                                                    cycle=False)

optimizer = optim.AdamWarmup(learning_rate=0.01,decay_steps=15000)

# 构建损失函数
mask_sparse_categotical_loss = Losess.MaskSparseCategoricalCrossentropy(from_logits=False)

f1score = Metric.SparseF1Score(average="macro")
precsionscore = Metric.SparsePrecisionScore(average="macro")
recallscore = Metric.SparseRecallScore(average="macro")
accuarcyscore = Metric.SparseAccuracy()

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory="./save",
                                     checkpoint_name="model.ckpt",
                                     max_to_keep=3)

Batch = 0
for X, token_type_id, input_mask, Y in load.load_train():
    with tf.GradientTape() as tape:
        predict = model(X)
        loss = mask_sparse_categotical_loss(Y, predict)
        f1 = f1score(Y, predict)
        precision = precsionscore(Y, predict)
        recall = recallscore(Y, predict)
        accuracy = accuarcyscore(Y, predict)
        if Batch % 20 == 0:
            print("Batch:{}\tloss:{:.4f}".format(Batch, loss.numpy()))
            print("Batch:{}\tacc:{:.4f}".format(Batch, accuracy))
            print("Batch:{}\tprecision{:.4f}".format(Batch, precision))
            print("Batch:{}\trecall:{:.4f}".format(Batch, recall))
            print("Batch:{}\tf1score:{:.4f}".format(Batch, f1))
        if Batch % 10:
            manager.save(checkpoint_number=Batch)
    grads_bert = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads_bert, model.trainable_variables))
    Batch += 1
