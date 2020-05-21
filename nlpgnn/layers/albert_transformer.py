#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from nlpgnn.tools import create_initializer
from .attention import ALBERTAttention
from nlpgnn.layers import dense


class AlbertTransformer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=1,
                 attention_head_size=64,
                 attention_probs_dropout_prob=0.0,
                 intermediate_size=3072,
                 intermediate_act_fn=None,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.0,
                 use_einsum=True,
                 name=None,
                 **kwargs):
        super(AlbertTransformer, self).__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.use_einsum = use_einsum
        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        self.attention = ALBERTAttention(
            num_attention_heads=self.num_attention_heads,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            use_einsum=True,
            name='self',
        )

        self.dense_layer_3d_proj = dense.DenseLayer3dProj(
            self.hidden_size,
            self.attention_head_size,
            create_initializer(self.initializer_range),
            None,
            use_einsum=self.use_einsum,
            name="dense"
        )

        self.dense_layer_2d = dense.DenseLayer2d(
            self.intermediate_size,
            create_initializer(self.initializer_range),
            self.intermediate_act_fn,
            use_einsum=self.use_einsum,
            num_attention_heads=self.num_attention_heads,
            name="dense"
        )

        self.out_dense_layer_2d = dense.DenseLayer2d(
            self.hidden_size,
            create_initializer(self.initializer_range),
            None,
            use_einsum=self.use_einsum,
            num_attention_heads=self.num_attention_heads,
            name="dense"
        )
        self.attdropout = tf.keras.layers.Dropout(self.hidden_dropout_prob)
        self.ffdropout = tf.keras.layers.Dropout(self.hidden_dropout_prob)
        self.attlayer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="LayerNorm")
        self.ffnlayer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="LayerNorm")

        self.built = True

    def call(self, input_tensor, attention_mask=None, is_training=True):
        with tf.keras.backend.name_scope("attention_1"):
            attention_output = self.attention(input_tensor, input_tensor,
                                              attention_mask, True)
            with tf.keras.backend.name_scope("output"):
                attention_output = self.dense_layer_3d_proj(attention_output)
                attention_output = self.attdropout(attention_output, training=is_training)
        attention_output = self.attlayer_norm(attention_output + input_tensor)
        with tf.keras.backend.name_scope("ffn_1"):
            with tf.keras.backend.name_scope("intermediate"):
                intermediate_output = self.dense_layer_2d(attention_output)
                with tf.keras.backend.name_scope("output"):
                    ffn_output = self.out_dense_layer_2d(intermediate_output)
                ffn_output = self.ffdropout(ffn_output, training=is_training)
        ffn_output = self.ffnlayer_norm(ffn_output + attention_output)
        return ffn_output
