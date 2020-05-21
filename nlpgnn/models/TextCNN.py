#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
import numpy as np


class TextCNN(tf.keras.Model):
    def __init__(self,
                 maxlen,
                 vocab_size,
                 embedding_dims,
                 class_num=2,
                 weights=None,
                 weights_trainable=False,
                 kernel_sizes=[2, 3, 4],
                 filter_size=[128, 128, 128],
                 name=None,
                 **kwargs):
        super(TextCNN, self).__init__(name=name, **kwargs)
        self.maxlen = maxlen

        self.kernel_sizes = kernel_sizes
        # "mask_zero: 是否把 0 看作为一个应该被遮蔽的特殊的 "padding" 值。
        # 这对于可变长的 循环神经网络层 十分有用。 如果设定为 True，
        # 那么接下来的所有层都必须支持 masking，否则就会抛出异常。
        # 如果 mask_zero 为 True，作为结果，索引 0 就不能被用于词汇表中
        # （input_dim 应该与 vocabulary + 1 大小相同）。"
        if weights != None:
            weights = np.array(weights)
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dims,
                                                       input_length=maxlen, weights=[weights],
                                                       trainable=weights_trainable)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dims,
                                                       input_length=maxlen)

        self.convs = []
        self.max_poolings = []
        for i, k in enumerate(kernel_sizes):
            # Conv1D inputs=[None,maxlen,embedding_dim]
            self.convs.append(tf.keras.layers.Conv1D(filter_size[i],
                                                     k,
                                                     activation="relu"))
            # here a little different from methods provide in article,
            # but they have the same purpose
            self.max_poolings.append(tf.keras.layers.GlobalAvgPool1D())
        self.dense = tf.keras.layers.Dense(class_num, activation='softmax')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        embedding = self.embedding(inputs)
        convs = []
        # embedding = self.bn(embedding, training=training)
        for i, k in enumerate(self.kernel_sizes):
            out = self.convs[i](embedding)
            out = self.max_poolings[i](out)
            convs.append(out)
        out = tf.keras.layers.concatenate(convs)
        out = self.bn(out, training=training)
        out = self.dense(out)
        return out

    def predict(self, inputs, training=False):
        re = self(inputs, training)
        return re
