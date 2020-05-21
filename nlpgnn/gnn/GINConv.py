#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""

import tensorflow as tf

from nlpgnn.gnn.messagepassing import MessagePassing
from nlpgnn.gnn.utils import GNNInput, remove_self_loop


class GINConvolution(MessagePassing):
    def __init__(self,
                 nn,
                 eps=0,
                 train_eps=False,
                 **kwargs):
        super(GINConvolution, self).__init__(aggr="sum", **kwargs)
        self.nn = nn
        self.eps = eps
        self.train_eps = train_eps

    def build(self, input_shapes):
        if self.train_eps:
            self.eps = self.add_weight(
                shape=(1,),
                initializer=tf.constant_initializer([self.eps]),
                name='eps',
            )
        else:
            self.eps = self.eps
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
        messages = edge_source_states
        return messages

    def call(self, inputs, training):
        adjacency_lists = inputs.adjacency_lists
        node_embeddings = inputs.node_embeddings
        part1 = (1 + self.eps) * node_embeddings
        part2 = self.propagate(GNNInput(node_embeddings, adjacency_lists), training)
        out = self.nn(part1 + part2, training=training)
        return out
