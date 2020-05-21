#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from nlpgnn.tools import get_activation, get_shape_list, create_initializer
from nlpgnn.layers import dense
from nlpgnn.layers.embedding import WDEmbedding, SegPosEmbedding
from nlpgnn.layers.albert_transformer import AlbertTransformer


class ALBERT(tf.keras.layers.Layer):
    def __init__(self,
                 param=None,
                 batch_size=2,
                 maxlen=128,
                 vocab_size=30000,
                 hidden_size=4096,
                 hidden_act="gelu",
                 num_hidden_groups=1,
                 embedding_size=128,
                 inner_group_num=1,
                 num_attention_heads=64,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.,
                 type_vocab_size=2,
                 intermediate_size=3072,
                 max_position_embeddings=512,
                 num_hidden_layers=12,
                 attention_probs_dropout_prob=0.,
                 use_one_hot_embedding=False,
                 use_einsum=True,
                 name=None,
                 **kwargs):
        super(ALBERT, self).__init__(name=name, **kwargs)
        self.maxlen = param.get("maxlen", maxlen)
        self.intermediate_size = param.get("intermediate_size", intermediate_size)
        self.vocab_size = param.get("vocab_size", vocab_size)
        self.batch_size = param.get("batch_size", batch_size)
        self.hidden_size = param.get("hidden_size", hidden_size)
        self.hidden_act = param.get("hidden_act", hidden_act)
        self.initializer_range = param.get("initializer_range", initializer_range)
        self.hidden_dropout_prob = param.get("hidden_dropout_prob", hidden_dropout_prob)
        self.type_vocab_size = param.get("type_vocab_size", type_vocab_size)
        self.num_attention_heads = param.get("num_attention_heads", num_attention_heads)
        self.max_position_embeddings = param.get("max_position_embeddings", max_position_embeddings)
        self.attention_probs_dropout_prob = param.get("attention_probs_dropout_prob", attention_probs_dropout_prob)
        self.num_hidden_layers = param.get("num_hidden_layers", num_hidden_layers)
        self.num_hidden_groups = param.get("num_hidden_groups", num_hidden_groups)
        self.inner_group_num = param.get("inner_group_num", inner_group_num)
        self.embedding_size = param.get("embedding_size", embedding_size)
        self.use_einsum = use_einsum
        self.use_one_hot_embedding = use_one_hot_embedding
        self.attention_head_size = hidden_size // num_attention_heads

    def build(self, input_shape):
        self.token_embedding = WDEmbedding(vocab_size=self.vocab_size,
                                           embedding_size=self.embedding_size,
                                           initializer_range=self.initializer_range,
                                           word_embedding_name="word_embeddings",
                                           use_one_hot_embedding=self.use_one_hot_embedding,
                                           name="embeddings")
        # segment and position embedding
        self.segposembedding = SegPosEmbedding(use_token_type=True,
                                               hidden_dropout_prob=self.hidden_dropout_prob,
                                               token_type_vocab_size=self.type_vocab_size,
                                               token_type_embedding_name="token_type_embeddings",
                                               use_position_embeddings=True,
                                               position_embedding_name="position_embeddings",
                                               initializer_range=self.initializer_range,
                                               max_position_embeddings=self.max_position_embeddings,
                                               use_one_hot_embedding=self.use_one_hot_embedding,
                                               name="embeddings"
                                               )
        self.shape_change = dense.DenseLayer2d(
            self.hidden_size,
            create_initializer(self.initializer_range),
            None,
            use_einsum=self.use_einsum,
            name="embedding_hidden_mapping_in",
        )

        self.encoder_layer = AlbertTransformer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            attention_head_size=self.attention_head_size,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            intermediate_size=self.intermediate_size,
            intermediate_act_fn=get_activation(self.hidden_act),
            initializer_range=self.initializer_range,
            hidden_dropout_prob=self.hidden_dropout_prob,
            use_einsum=True,
            name="inner_group_{}".format(0)
        )

        self.pool_out = tf.keras.layers.Dense(
            self.hidden_size,
            activation=tf.tanh,
            # kernel_constraint=create_initializer(self.initializer_range),
            name="dense")
        self.built = True

    def call(self, inputs, is_training=True):
        input_ids, token_type_ids, input_mask = tf.split(inputs, 3, 0)
        input_ids = tf.cast(tf.squeeze(input_ids, axis=0), tf.int32)
        token_type_ids = tf.cast(tf.squeeze(token_type_ids, axis=0), tf.int32)
        input_mask = tf.cast(tf.squeeze(input_mask, axis=0), tf.int32)
        input_shape = get_shape_list(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        with tf.keras.backend.name_scope("bert"):
            self.embedding_output = self.token_embedding(input_ids)
            self.embedding_output = self.segposembedding(self.embedding_output, token_type_ids, is_training)
            with tf.keras.backend.name_scope("encoder"):
                input_shape = get_shape_list(self.embedding_output, expected_rank=3)
                input_width = input_shape[2]
                self.all_layer_outputs = []
                if input_width != self.hidden_size:
                    prev_output = self.shape_change(self.embedding_output)
                else:
                    prev_output = self.embedding_output
                with tf.keras.backend.name_scope("transformer"):
                    for i in range(self.num_hidden_layers):
                        group_idx = int(i / self.num_hidden_layers * self.num_hidden_groups)

                        with tf.keras.backend.name_scope("group_%d" % group_idx):
                            layer_output = prev_output

                            for inner_group_idx in range(self.inner_group_num):
                                # with tf.keras.backend.name_scope("layer_%d" % i):
                                # for encoder_layer in encoder_layers:
                                layer_output = self.encoder_layer(layer_output, input_mask, is_training)
                                prev_output = layer_output
                                self.all_layer_outputs.append(layer_output)
            self.sequence_output = layer_output

        return self

    def get_pooled_output(self):
        with tf.keras.backend.name_scope("pooler"):
            first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
            self.pooled_output = self.pool_out(first_token_tensor)
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_layer_outputs

    def get_embedding_output(self):
        return self.embedding_output
