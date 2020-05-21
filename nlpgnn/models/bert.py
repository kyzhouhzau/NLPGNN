#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from nlpgnn.tools import get_activation, get_shape_list, create_attention_mask_from_input_mask
from nlpgnn.layers.embedding import WDEmbedding, SegPosEmbedding
from nlpgnn.layers.transformer import Transformer

class BERT(tf.keras.layers.Layer):
    def __init__(self,
                 param=None,
                 batch_size=2,
                 maxlen=128,
                 vocab_size=21128,
                 hidden_size=768,
                 hidden_act="gelu",
                 num_attention_heads=12,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 type_vocab_size=2,
                 intermediate_size=3072,
                 max_position_embeddings=512,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=True,
                 num_hidden_layers=12,
                 name=None,
                 **kwargs):
        super(BERT, self).__init__(name=name, **kwargs)
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

        self.use_one_hot_embeddings = use_one_hot_embeddings

    def build(self, input_shape):
        self.token_embedding = WDEmbedding(vocab_size=self.vocab_size,
                                           embedding_size=self.hidden_size,
                                           initializer_range=self.initializer_range,
                                           word_embedding_name="word_embeddings",
                                           use_one_hot_embedding=self.use_one_hot_embeddings,
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
                                               use_one_hot_embedding=self.use_one_hot_embeddings,
                                               name="embeddings"
                                               )
        self.encoder_layers = []
        for layer_idx in range(self.num_hidden_layers):
            self.encoder_layer = Transformer(
                batch_size=self.batch_size,
                seq_length=self.maxlen,
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                intermediate_act_fn=get_activation(self.hidden_act),
                hidden_dropout_prob=self.attention_probs_dropout_prob,
                initializer_range=self.initializer_range,
                name="layer_{}".format(layer_idx)
            )
            self.encoder_layers.append(self.encoder_layer)

        self.pool_out = tf.keras.layers.Dense(
            self.hidden_size,
            activation=tf.tanh,
            # kernel_constraint=create_initializer(config.initializer_range),
            name="dense"
        )

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
        self.embedding_output = self.token_embedding(input_ids)
        self.embedding_output = self.segposembedding(self.embedding_output, token_type_ids, is_training)
        with tf.keras.backend.name_scope("encoder"):
            attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)
            self.all_layer_outputs = []
            layer_encode_output = self.embedding_output
            # print(layer_encode_output)#[3,5,768]
            for encoder_layer in self.encoder_layers:
                layer_encode_input = layer_encode_output
                layer_encode_output = encoder_layer(layer_encode_input, attention_mask, is_training)
                self.all_layer_outputs.append(layer_encode_output)
            self.sequence_output = layer_encode_output
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
