#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf
from typing import NamedTuple, List
from nlpgnn.gnn.utils import batch_read_out
from nlpgnn.gnn.TGCNConv import TextGCNConvolution


class GNNInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: List
    edge_weights: List
    etan: tf.Tensor


class TextGCN2019(tf.keras.Model):
    def __init__(self, dim, num_class, drop_rate, **kwargs):
        super(TextGCN2019, self).__init__(**kwargs)
        self.conv1 = TextGCNConvolution()
        self.dense1 = tf.keras.layers.Dense(dim)
        self.dense2 = tf.keras.layers.Dense(num_class)
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        self.drop2 = tf.keras.layers.Dropout(drop_rate)
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.norm2 = tf.keras.layers.BatchNormalization()

    def call(self, node_embeddings, etans, edge_indexs, batchs, edge_weights, training=True):
        edge_indexs = [edge_indexs]
        edge_weights = [edge_weights]
        x = tf.nn.relu(self.conv1(GNNInput(node_embeddings, edge_indexs, edge_weights, etans), training))
        x = self.norm1(x, training=training)
        x = batch_read_out(x, batchs)
        x = tf.nn.relu(self.dense1(x))
        x = self.norm2(x, training=training)
        x = self.drop1(x, training=training)
        x = self.dense2(x)
        x = self.drop2(x, training=training)
        return tf.math.softmax(x, -1)

    def predict(self, node_embeddings, etans, edge_indexs, batchs, edge_weights, training=False):
        return self(node_embeddings, etans, edge_indexs, batchs, edge_weights, training)
