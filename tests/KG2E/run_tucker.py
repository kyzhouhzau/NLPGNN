#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import numpy as np
import tensorflow as tf

from nlpgnn.datas.graphloader import TuckERLoader
from nlpgnn.metrics import Metric
from nlpgnn.models import tucker

lr = 0.005
label_smoothing = 0.1
batch_size = 128
training = True

loader = TuckERLoader(base_path="data")
er_vocab, er_vocab_pairs = loader.data_dump("train")

evaluate = Metric.HitN_MR_MRR(loader, mode="valid")

model = tucker.TuckER(loader)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=2000, decay_rate=0.995)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# 构建损失函数
binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=label_smoothing)

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory="./save",
                                     checkpoint_name="model.ckpt",
                                     max_to_keep=3)

for epoch in range(500):
    losses = []
    Batch = 0
    for data in loader.get_batch(er_vocab, er_vocab_pairs, batch_size=batch_size):
        with tf.GradientTape() as tape:
            h_idx = data['h_r'][:, 0]
            r_idx = data['h_r'][:, 1]
            t_idx = data['t']
            predict = model(h_idx, r_idx, training)
            targets = loader.target_convert(t_idx, batch_size, len(loader.entities))
            loss = binary_loss(targets, predict)
            loss = tf.reduce_mean(loss)
            losses.append(loss.numpy())
            hit1, hit3, hit5, hit10, MR, MRR = evaluate(model, batch_size)
            print("Epoch:{}\tLoss:{:.4f}\tHit@5:{:.4f}\tHit@10:{:.4f}\tMRR{:.4f}\n".format(epoch, np.mean(losses),
                                                                                           hit5, hit10, MRR))
        # For filter Warning in calculate gradient of moving_mean and moving_variance
        grads = tape.gradient(loss, model.trainable_variable)
        optimizer.apply_gradients(
            grads_and_vars=zip(grads, model.trainable_variable))
        Batch += 1

