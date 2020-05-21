#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf

from nlpgnn.tools import create_initializer, get_activation
from .attention import GPTAttention


class MLP(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, initializer_range,
                 resid_pdrop_rate, name='mlp', **kwargs):
        super(MLP, self).__init__(name, **kwargs)
        self.resid_pdrop_rate = resid_pdrop_rate
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.c_fc = tf.keras.layers.Dense(
            hidden_size * 4,
            name="c_fc",
            kernel_initializer=create_initializer(self.initializer_range)
        )

        self.c_proj = tf.keras.layers.Dense(
            hidden_size,
            name="c_proj",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        self.act = get_activation('gelu')

        self.dropout = tf.keras.layers.Dropout(self.resid_pdrop_rate)

    def call(self, input, training=True):
        h = self.act(self.c_fc(input))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2


class GPT2Transformer(tf.keras.layers.Layer):
    def __init__(self,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 epsilon=1e-8,
                 resid_out_rate=0.0,
                 name=None,
                 **kwargs):
        super(GPT2Transformer, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.resid_out_rate = resid_out_rate

    def build(self, input_shape):
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                       name='ln_1')
        # self.ln_1 = normalization.GPTNorm(epsilon=self.epsilon,
        #                                                name='ln_1')
        self.attn = GPTAttention(
            num_attention_heads=self.num_attention_heads,
            resid_out_rate=self.resid_out_rate,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            scale=True,
            name="attn")

        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                       name='ln_2')
        # self.ln_2 = normalization.GPTNorm(epsilon=self.epsilon,
        #                                                name='ln_2')
        self.mlp = MLP(self.num_attention_heads, self.initializer_range,
                       self.resid_out_rate, name='mlp')
        self.built = True

    def call(self, input, past=None, training=False):
        layer_norm_output = self.ln_1(input)
        output_attn, present = self.attn(layer_norm_output, past, training=training)
        residual_output = input + output_attn
        layer_norm_output = self.ln_2(residual_output)
        mlp_output = self.mlp(layer_norm_output, training=training)
        residual_output = mlp_output + residual_output
        return residual_output, present  # x, present
