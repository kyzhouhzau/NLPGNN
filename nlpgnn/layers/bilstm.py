#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
import numpy as np


class BiLSTM(tf.keras.layers.Layer):
    def __init__(self, maxlen,
                 vocab_size,
                 embedding_dims,
                 hidden_dim,
                 dropout_rate=0.0,
                 return_state=False,
                 return_sequences=True,
                 weights=None,
                 weights_trainable=True, **kwargs):
        super(BiLSTM, self).__init__(**kwargs)
        if weights != None:
            weights = np.array(weights)
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dims,
                                                       input_length=maxlen, weights=[weights],
                                                       trainable=weights_trainable)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dims,
                                                       input_length=maxlen)
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_dim, return_state=return_state,
                                 return_sequences=return_sequences, dropout=dropout_rate)
        )

    def call(self, inputs, training):
        embed = self.embedding(inputs)
        logits = self.bilstm(embed, training=training)
        return logits

    def predict(self, inputs, training):
        logits = self(inputs, training)
        return logits
