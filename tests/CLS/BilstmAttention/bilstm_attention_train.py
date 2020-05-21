# ! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf
from nlpgnn.layers import bilstm, attention
from nlpgnn.optimizers import optim
from nlpgnn.metrics import Metric, Losess
from nlpgnn.datas.dataloader import TFWriter, TFLoader

maxlen = 128
batch_size = 64
embedding_dims = 100
hidden_dim = 50
vocab_file = "Input/vocab.txt"
vocab_size = 30522  # line in vocab.txt
class_num = 2

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(maxlen, vocab_file, modes=["train"], task='cls', check_exist=False)

load = TFLoader(maxlen, batch_size, task='cls', epoch=3)


class BilstmAttention(tf.keras.Model):
    def __init__(self, maxlen,
                 vocab_size,
                 embedding_dims,
                 hidden_dim,
                 dropout_rate=0.5,
                 return_state=False,
                 return_sequences=True,
                 weights=None, **kwargs):
        super(BilstmAttention, self).__init__(**kwargs)
        self.bilstm = bilstm.BiLSTM(maxlen, vocab_size,
                                    embedding_dims, hidden_dim,
                                    dropout_rate=dropout_rate, return_state=return_state,
                                    return_sequences=return_sequences, weights=weights)
        self.dense = tf.keras.layers.Dense(class_num, activation='softmax')
        self.att = attention.HieAttention(2 * hidden_dim, attention_size=100)

    def call(self, inputs, training=True):
        logits = self.bilstm(inputs, training)  # [B,T,2H]
        logits, _ = self.att(logits)
        logits = self.dense(logits)
        return logits

    def predict(self,inputs,training=False):
        out = self(inputs,training)
        return out



model = BilstmAttention(maxlen, vocab_size, embedding_dims, hidden_dim)

# 构建优化器
optimizer = optim.AdamWarmup(0.01, decay_steps=15000)
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
        if Batch % 10 == 0:
            print("Batch:{}\tloss:{:.4f}".format(Batch, loss.numpy()))
            print("Batch:{}\tacc:{:.4f}".format(Batch, accuracy))
            print("Batch:{}\tprecision{:.4f}".format(Batch, precision))
            print("Batch:{}\trecall:{:.4f}".format(Batch, recall))
            print("Batch:{}\tf1score:{:.4f}".format(Batch, f1))
            manager.save(checkpoint_number=Batch)
    grads_bert = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads_bert, model.variables))
    Batch += 1
