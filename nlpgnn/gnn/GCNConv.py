#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""

import tensorflow as tf

from nlpgnn.gnn.messagepassing import MessagePassing
from nlpgnn.gnn.utils import GNNInput


class GraphConvolution(MessagePassing):
    def __init__(self,
                 out_features,
                 aggr="sum",
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 regularizer=None,
                 use_bias=False,
                 cached=True,
                 **kwargs):
        super(GraphConvolution, self).__init__(aggr, **kwargs)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.regularizer = tf.keras.initializers.get(regularizer)
        self.use_bias = use_bias
        self.out_features = out_features
        self.cached = cached

    def build(self, input_shapes):
        node_embedding_shapes = input_shapes.node_embeddings
        # adjacency_list_shapes = input_shapes.adjacency_lists
        in_features = node_embedding_shapes[-1]
        self.weight = self.add_weight(
            shape=(in_features, self.out_features),
            initializer=self.kernel_initializer,
            regularizer=self.regularizer,
            name='wt',
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_features,),
                initializer=self.bias_initializer,
                name='b',
            )

        self.cached_result = None
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
        if not self.cached or self.cached_result is None:
            deg_target = tf.math.pow(num_incoming_to_node_per_message, -0.5)  # [M]
            deg_source = tf.math.pow(num_outing_to_node_per_message, -0.5)  # [M]
            norm = deg_source * deg_target
            self.cached_result = norm
        norm = self.cached_result
        messages = tf.reshape(norm, [-1, 1]) * edge_source_states
        return messages

    def call(self, inputs):
        adjacency_lists = inputs.adjacency_lists
        node_embeddings = inputs.node_embeddings
        if self.use_bias:
            node_embeddings = tf.linalg.matmul(node_embeddings, self.weight) + self.bias
        else:
            node_embeddings = tf.linalg.matmul(node_embeddings, self.weight)
        aggr_out = self.propagate(GNNInput(node_embeddings, adjacency_lists))
        return aggr_out
