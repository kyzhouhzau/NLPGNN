#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
# Text Level Graph Neural Network for Text Classfication
"""

import tensorflow as tf

from nlpgnn.gnn.messagepassing import MessagePassing
from nlpgnn.gnn.utils import *


class GNNInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: List
    edge_weights: List
    # etan: tf.Tensor


class TextGCNConvolution(MessagePassing):
    def __init__(self,
                 aggr='max',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(TextGCNConvolution, self).__init__(aggr=aggr, **kwargs)
        self.kernel_initializer = kernel_initializer

    # def build(self, input_shapes):
    #     node_embedding_shapes = input_shapes.node_embeddings
    #     adjacency_list_shapes = input_shapes.adjacency_lists
    #     in_features = node_embedding_shapes[0]
    #     self.eta = self.add_weight(
    #         shape=(1,),
    #         initializer=tf.constant_initializer(0.5),
    #         name='eta'
    #     )
    #
    #     self.built = True

    def message_function(self, edge_source_states, edge_source,  # x_j source
                         edge_target_states, edge_target,  # x_i target
                         num_incoming_to_node_per_message,  # degree target
                         num_outing_to_node_per_message,  # degree source
                         edge_weights,
                         edge_type_idx, training):
        """
        :param edge_source_states: [M,H]
        :param edge_target_states: [M,H]
        :param num_incoming_to_node_per_message:[M]
        :param edge_type_idx:
        :param training:
        :return:
        """
        edge_weight = edge_weights[edge_type_idx]

        return edge_source_states * tf.reshape(edge_weight, [-1, 1])
        # return edge_source_states

    def propagate(self, inputs, training=None):
        """
        N: Num of nodes
        D: Dim of nodes features
        E: Num of edges
        :param node_embeddings: [N,D]
        :param adjacency_lists: [E,2]
        :return:
        """

        node_embeddings = inputs.node_embeddings
        adjacency_lists = inputs.adjacency_lists
        edge_weight_lists = inputs.edge_weights
        nodes_num = tf.shape(node_embeddings)[0]

        # 所有类别中每一个节点收到的信息大小 shape=[E*type_num,D]
        messages_all_type = self._calculate_messages_all_type(node_embeddings, adjacency_lists, edge_weight_lists,
                                                              training)

        # 所有关系类别中信息流动的目标节点 shape=[E]
        edge_type_to_message_targets = [adjacency_type_list[:, 1] for adjacency_type_list in adjacency_lists]

        # 就当前节点聚合所有收到的信息，并获得新的节点状态
        new_nodes_states = self._aggregate_function(messages_all_type,
                                                    edge_type_to_message_targets,
                                                    nodes_num)

        return new_nodes_states

    def _calculate_messages_all_type(self, node_embeddings, adjacency_lists, edge_weights, training):
        messages_all_type = []
        type_incoming_edges_num, type_outing_edges_num = self._calculate_type_to_incoming_edges_num(node_embeddings,
                                                                                                    adjacency_lists)
        for edge_type_idx, adjanceny_list_edge_type in enumerate(adjacency_lists):  # M表示当前类的edge类型中包含的edge个数
            edge_sources = adjanceny_list_edge_type[:, 0]  # [M]
            edge_targets = adjanceny_list_edge_type[:, 1]  # [M]
            edge_source_states = tf.gather(node_embeddings, edge_sources)  # [M,D]
            edge_targets_states = tf.gather(node_embeddings, edge_targets)  # [M,D]
            # print(edge_source_states[6955])
            # print(edge_sources[6955])
            # print(node_embeddings[2489])
            num_incoming_to_node_per_message = tf.gather(
                type_incoming_edges_num[edge_type_idx, :], edge_targets)  # message num [M], 目标每一个节点的message输入
            num_outing_to_node_per_message = tf.gather(
                type_outing_edges_num[edge_type_idx, :], edge_sources)

            messages = self.message_function(edge_source_states, edge_sources,
                                             edge_targets_states, edge_targets,
                                             num_incoming_to_node_per_message,
                                             num_outing_to_node_per_message,
                                             edge_weights,
                                             edge_type_idx, training)
            messages_all_type.append(messages)
        return messages_all_type

    def call(self, inputs, training):
        adjacency_lists = inputs.adjacency_lists
        node_embeddings = inputs.node_embeddings
        edge_weights_lists = inputs.edge_weights
        etan = inputs.etan
        # adjacency_lists = add_remain_self_loop(adjacency_lists, len(node_embeddings))
        aggr_out = self.propagate(GNNInput(node_embeddings, adjacency_lists, edge_weights_lists), training)

        aggr_out = (1 - etan) * aggr_out + etan * node_embeddings

        return aggr_out
