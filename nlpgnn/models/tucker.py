#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""

import tensorflow as tf

import numpy as np



class TuckER(tf.keras.Model):
    # loader.data
    def __init__(self, loader,
                 ent_vec_dim=200,
                 rel_vec_dim=30,
                 input_dropout=0.2,
                 hidden_dropout1=0.1,
                 hidden_dropout2=0.2,
                 **kwargs):
        super(TuckER, self).__init__(**kwargs)
        self.rel_vec_dim = rel_vec_dim
        self.ent_vec_dim = ent_vec_dim
        self.E = tf.keras.layers.Embedding(len(loader.entities),
                                           ent_vec_dim)
        self.R = tf.keras.layers.Embedding(len(loader.relations),
                                           rel_vec_dim)

        self.input_dropout = tf.keras.layers.Dropout(input_dropout)
        self.hidden_dropout1 = tf.keras.layers.Dropout(hidden_dropout1)
        self.hidden_dropout2 = tf.keras.layers.Dropout(hidden_dropout2)

        self.bn0 = tf.keras.layers.BatchNormalization()
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.W = tf.Variable(
            initial_value=tf.constant(
                np.random.uniform(-1, 1, (self.rel_vec_dim, self.ent_vec_dim, self.ent_vec_dim)
                                  ), tf.float32),
            trainable=True,
        )

    def call(self, e1_idx, r_idx, training=True):
        e1 = self.E(e1_idx)
        x = self.bn0(e1, training=training)
        x = self.input_dropout(x, training=training)
        x = tf.reshape(x, [-1, 1, e1.shape[1]])

        r = self.R(r_idx)
        W_mat = tf.matmul(r, tf.reshape(self.W, shape=[r.shape[1], -1]))
        W_mat = tf.reshape(W_mat, shape=[-1, e1.shape[1], e1.shape[1]])
        W_mat = self.hidden_dropout1(W_mat, training=training)
        x = tf.matmul(x, W_mat)
        x = tf.reshape(x, shape=[-1, e1.shape[1]])
        x = self.bn1(x, training=training)
        x = self.hidden_dropout2(x, training=training)
        x = tf.matmul(x, tf.transpose(tf.constant(self.E.get_weights()[0]), [1, 0]))
        pred = tf.math.sigmoid(x)
        return pred

    def predict(self, e1_idx, r_idx):
        pre = self(e1_idx, r_idx, training=False)
        return pre

    def get_config(self):
        config = {
            'ent_vec_dim': self.ent_vec_dim,
            "rel_vec_dim": self.rel_vec_dim,
            'input_dropout': self.input_dropout,
            "hidden_dropout1": self.hidden_dropout1,
            "hidden_dropout2": self.hidden_dropout2,
        }
        base_config = super(TuckER, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
