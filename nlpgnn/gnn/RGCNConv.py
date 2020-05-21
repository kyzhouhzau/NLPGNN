#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
Usage:
node_embeddings = tf.random.normal(shape=(5, 3))
adjacency_lists = [
                    tf.constant([[0, 1], [2, 4], [2, 4]], dtype=tf.int32),
                    tf.constant([[0, 1], [2, 4], [2, 4]], dtype=tf.int32)
                  ]
layer = RGraphConvolution(out_features=12)
x = layer(GNNInput(node_embeddings, adjacency_lists))
"""
import tensorflow as tf

from nlpgnn.gnn.messagepassing import MessagePassing


class RGraphConvolution(MessagePassing):
    def __init__(self,
                 out_features,
                 epsion=1e-7,
                 aggr="sum",
                 normalize=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_bias=True,
                 **kwargs):
        super(RGraphConvolution, self).__init__(aggr, **kwargs)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.use_bias = use_bias
        self.normalize = normalize
        self.out_features = out_features
        self.epsion = epsion

    def build(self, input_shapes):
        node_embedding_shapes = input_shapes.node_embeddings
        adjacency_list_shapes = input_shapes.adjacency_lists
        num_edge_type = len(adjacency_list_shapes)
        in_features = node_embedding_shapes[-1]
        self._edge_type_weights = []
        self._edge_type_bias = []
        for i in range(num_edge_type):
            weight = self.add_weight(
                shape=(in_features, self.out_features),
                initializer=self.kernel_initializer,
                name='wt_{}'.format(i),
            )
            self._edge_type_weights.append(weight)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_features),
                initializer=self.bias_initializer,
                name='b',
            )
        else:
            self.bias = None

        self.weight_o = self.add_weight(
            shape=(in_features, self.out_features),
            initializer=self.kernel_initializer,
            name='wo',
        )
        self.built = True

    def message_function(self, edge_source_states,
                         edge_target_states,
                         num_incoming_to_node_per_message,
                         num_outing_to_node_per_message,
                         edge_type_idx):
        """
        :param edge_source_states: [M,H]
        :param edge_target_states: [M,H]
        :param num_incoming_to_node_per_message:[M]
        :param edge_type_idx:
        :param training:
        :return:
        """
        weight_r = self._edge_type_weights[edge_type_idx]
        messages = tf.linalg.matmul(edge_source_states, weight_r)
        if self.normalize:
            messages = (
                    tf.expand_dims(1.0 / (tf.cast(num_incoming_to_node_per_message,
                                                  tf.float32) + self.epsion), axis=-1) * messages
            )
        return messages

    def call(self, inputs):
        aggr_out = self.propagate(inputs)  # message_passing + update
        aggr_out += tf.linalg.matmul(inputs.node_embeddings, self.weight_o)
        if self.bias is not None:
            aggr_out += self.bias
        return aggr_out
