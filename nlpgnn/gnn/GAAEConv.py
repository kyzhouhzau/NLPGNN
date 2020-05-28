#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""

import tensorflow as tf

from nlpgnn.gnn.messagepassing import MessagePassing
from nlpgnn.gnn.utils import GNNInput, masksoftmax


class GraphAttentionAutoEncoder(MessagePassing):
    def __init__(self,
                 out_features,
                 heads=1,
                 dropout_rate=0.,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 regularizer=5e-4,
                 concat=True,
                 **kwargs):
        super(GraphAttentionAutoEncoder, self).__init__(aggr="sum", **kwargs)
        self.use_bias = use_bias
        self.out_features = out_features
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer = regularizer
        self.concat = concat

    def build(self, input_shapes):
        node_embedding_shapes = input_shapes.node_embeddings
        # adjacency_list_shapes = input_shapes.adjacency_lists
        in_features = node_embedding_shapes[-1]

        self.att = self.add_weight(
            shape=(1, self.heads, 2 * self.out_features),
            initializer=self.kernel_initializer,
            name='att',
        )

        if self.use_bias and self.concat:
            self.bias = self.add_weight(
                shape=(self.heads * self.out_features,),
                initializer=self.bias_initializer,
                name='b',
            )
        elif self.use_bias and not self.concat:
            self.bias = self.add_weight(
                shape=(self.out_features,),
                initializer=self.bias_initializer,
                name='b',
            )

        self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.built = True

    def message_function(self, edge_source_states, edge_source,  # x_j source
                         edge_target_states, edge_target,  # x_i target
                         num_incoming_to_node_per_message,  # degree target
                         num_outing_to_node_per_message,  # degree source
                         edge_type_idx, training):
        """
        :param edge_source_states: [M,H]
        :param edge_target_states: [M,H]
        :param num_incoming_to_node_per_message:[M]
        :param edge_type_idx:
        :param training:
        :return:
        """
        # 计算注意力系数
        alpha = tf.concat([edge_target_states, edge_source_states], -1) * self.att #[M,heads,2D]
        alpha = tf.reduce_sum(alpha, -1)  # [M,Head]
        alpha = tf.math.sigmoid(alpha)
        alpha = masksoftmax(alpha, edge_target)
        # alpha = self.drop1(alpha, training=training)
        # edge_source_states = self.drop2(edge_source_states, training=training)
        # messages = tf.math.sigmoid(edge_source_states) * tf.reshape(alpha, [-1, self.heads, 1])
        messages = edge_source_states * tf.reshape(alpha, [-1, self.heads, 1])
        return messages

    def call(self, inputs, weight, transpose_b, training):
        adjacency_lists = inputs.adjacency_lists
        node_embeddings = inputs.node_embeddings
        node_embeddings = tf.linalg.matmul(node_embeddings, weight, transpose_b=transpose_b)

        node_embeddings = tf.reshape(node_embeddings, [node_embeddings.shape[0], self.heads, -1])
        aggr_out = self.propagate(GNNInput(node_embeddings, adjacency_lists), training)
        if self.concat is True:
            aggr_out = tf.reshape(aggr_out, [-1, self.heads * self.out_features])
        else:
            aggr_out = tf.reduce_mean(aggr_out, 1)
        if self.use_bias:
            aggr_out += self.bias
        return aggr_out
