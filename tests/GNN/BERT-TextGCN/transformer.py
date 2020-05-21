#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from nlpgnn.tools import create_initializer
from attention import MultiAttentionLayer
from nlpgnn.tools import reshape_to_matrix, get_shape_list, reshape_from_matrix


class Transformer(tf.keras.layers.Layer):
    def __init__(self,
                 # layer_idx=0,
                 batch_size,
                 seq_length,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 name=None,
                 **kwargs):
        super(Transformer, self).__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.attention_layer = MultiAttentionLayer(
            num_attention_heads=self.num_attention_heads,
            size_per_head=attention_head_size,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            do_return_2d_tensor=True,
            batch_size=self.batch_size,
            from_seq_length=self.seq_length,
            to_seq_length=self.seq_length,
            name="self"

        )
        self.attention_output_layer = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=create_initializer(self.initializer_range),
            name="dense"
        )
        self.inter_output = tf.keras.layers.Dense(
            self.intermediate_size,
            activation=self.intermediate_act_fn,
            kernel_initializer=create_initializer(self.initializer_range),
            name="dense"
        )
        self.layer_out = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=create_initializer(self.initializer_range),
            name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(self.hidden_dropout_prob)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="LayerNorm")
        self.out_layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="LayerNorm")

        self.built = True

    def call(self, input_tensor, attention_mask=None, is_training=True):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size must be the integer multiple of num attention heads"
            )
        input_shape = get_shape_list(input_tensor)
        input_tensor = reshape_to_matrix(input_tensor)  # [15,768]
        # print(attention_mask)  # [3,5,5]
        # --------------------------------------------------------------------------------------
        with tf.keras.backend.name_scope("attention"):
            attention_heads = []
            attention_probs = []
            attention_head, attention_prob = self.attention_layer(input_tensor, input_tensor, attention_mask)
            attention_heads.append(attention_head)
            attention_probs.append(attention_prob)
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
                attention_prob = attention_probs[0]
            else:
                attention_output = tf.concat(attention_heads, axis=-1)
                attention_prob = tf.concat(attention_probs, axis=-1)
            with tf.keras.backend.name_scope("output"):
                attention_output = self.attention_output_layer(attention_output)
                attention_output = self.dropout(attention_output, training=is_training)
                attention_output = self.out_layer_norm(attention_output + input_tensor)
        with tf.keras.backend.name_scope("intermediate"):
            intermediate_output = self.inter_output(attention_output)
        with tf.keras.backend.name_scope("output"):
            layer_output = self.layer_out(intermediate_output)
            layer_output = self.dropout(layer_output, training=is_training)
            layer_output = self.layer_norm(layer_output + attention_output)
        layer_output = reshape_from_matrix(layer_output, input_shape)
        return layer_output, attention_prob
