#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""

import tensorflow as tf

from nlpgnn.gnn.messagepassing import MessagePassing
from nlpgnn.gnn.utils import GNNInput, masksoftmax


class GraphAttentionConvolution(MessagePassing):
    def __init__(self,
                 out_features,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout_rate=0.,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 regularizer=5e-4,
                 **kwargs):
        super(GraphAttentionConvolution, self).__init__(aggr="sum", **kwargs)
        self.use_bias = use_bias
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer = regularizer

    def build(self, input_shapes):
        node_embedding_shapes = input_shapes.node_embeddings
        # adjacency_list_shapes = input_shapes.adjacency_lists
        in_features = node_embedding_shapes[-1]
        self.weight = self.add_weight(
            shape=(in_features, self.heads * self.out_features),
            initializer=self.kernel_initializer,
            # regularizer=tf.keras.regularizers.l2(self.regularizer),
            name='wt',
        )
        self.att = self.add_weight(
            shape=(1, self.heads, 2 * self.out_features),
            initializer=self.kernel_initializer,
            # regularizer=tf.keras.regularizers.l2(self.regularizer),
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
        edge_source_states = tf.reshape(edge_source_states, [-1, self.heads, self.out_features])  # [M,Head,dim]
        edge_target_states = tf.reshape(edge_target_states, [-1, self.heads, self.out_features])  # [M,Head,dim]
        # self.att=[1,heads,2*D]
        # [M,keads,2D] * [1,heads,2D]
        alpha = tf.concat([edge_target_states, edge_source_states], -1) * self.att
        alpha = tf.reduce_mean(alpha, -1)
        alpha = tf.nn.leaky_relu(alpha, self.negative_slope)
        alpha = masksoftmax(alpha, edge_target)  # here not provide nodes num, because we have add self loop at the beginning.
        alpha = self.drop1(alpha, training=training)
        edge_source_states = self.drop2(edge_source_states, training=training)
        messages = edge_source_states * tf.reshape(alpha, [-1, self.heads, 1])
        return messages

    def call(self, inputs, training):
        adjacency_lists = inputs.adjacency_lists
        node_embeddings = inputs.node_embeddings
        node_embeddings = tf.linalg.matmul(node_embeddings, self.weight)
        aggr_out = self.propagate(GNNInput(node_embeddings, adjacency_lists), training)
        if self.concat is True:
            aggr_out = tf.reshape(aggr_out, [-1, self.heads * self.out_features])
        else:
            aggr_out = tf.reduce_mean(aggr_out, 1)

        if self.use_bias:
            aggr_out += self.bias
        return aggr_out
