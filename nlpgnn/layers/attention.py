#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:kaiyinzhou
@Tensorflow 2.0
All of the following Code was follow Google BERT!
"""

from __future__ import absolute_import, division, print_function
import math
import tensorflow as tf
from nlpgnn.tools import create_initializer, reshape_to_matrix, get_shape_list
from nlpgnn.layers import dense


def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
    output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor


# BERT Attention
class MultiAttentionLayer(tf.keras.layers.Layer):
    """
    Performs multi-headed attention from `from_tensor` to `to_tensor`.
    """

    def __init__(self,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 num_attention_heads=1,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 batch_size=None,
                 from_seq_length=None,
                 to_seq_length=None,
                 name=None,
                 **kwargs):
        super(MultiAttentionLayer, self).__init__(name=name, **kwargs)
        self.size_per_head = size_per_head
        self.query_act = query_act
        self.key_act = key_act
        self.num_attention_heads = num_attention_heads
        self.value_act = value_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.batch_size = batch_size
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        # `query_layer` =[B*F, N*H]
        self._query_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=self.query_act,
            name="query",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        # `value_layer` = [B*T, N*H]
        self._key_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=self.key_act,
            name="key",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        # `query_layer` =[B*T, N*H]
        self._value_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=self.value_act,
            name="value",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        self.drop_out = tf.keras.layers.Dropout(self.attention_probs_dropout_prob)
        self.built = True

    def call(self, from_tensor, to_tensor=None, attention_mask=None, is_training=True):
        from_shape = from_tensor.shape.as_list()
        to_shape = to_tensor.shape.as_list()
        if len(from_shape) != len(to_shape):
            raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")
        if len(from_shape) == 3:
            self.batch_size = from_shape[0]
            self.from_seq_length = from_shape[1]
            self.to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (self.batch_size is None or self.from_seq_length is None or self.to_seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified."
                )
        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)
        query_layer = self._query_layer(from_tensor_2d)
        key_layer = self._key_layer(to_tensor_2d)
        value_layer = self._value_layer(to_tensor_2d)
        # [B,N,F,H]
        query_layer = transpose_for_scores(query_layer, self.batch_size, self.num_attention_heads,
                                           self.from_seq_length, self.size_per_head)
        # [B,N,T,H]
        key_layer = transpose_for_scores(key_layer, self.batch_size, self.num_attention_heads,
                                         self.to_seq_length, self.size_per_head)
        # attention_score = [B,N,F,T]
        attention_score = tf.linalg.matmul(query_layer, key_layer, transpose_b=True)
        attention_score = tf.math.multiply(attention_score, 1.0 / math.sqrt(float(self.size_per_head)))
        if attention_mask is not None:
            # [batch_size, from_seq_length, to_seq_length]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            # Here we convert aim position to zero, masked position to -10000
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            # This could let score for masked very low
            attention_score += adder
        # sofmax
        attention_probs = tf.math.softmax(attention_score)
        attention_probs = self.drop_out(attention_probs, training=is_training)
        # value [B,N,T,H]
        value_layer = transpose_for_scores(value_layer, self.batch_size, self.num_attention_heads,
                                           self.to_seq_length, self.size_per_head)
        # context_layer [B,N,T,H]
        context_layer = tf.linalg.matmul(attention_probs, value_layer)
        # [B,F,N,H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        if self.do_return_2d_tensor:
            context_layer = tf.reshape(context_layer, [self.batch_size * self.from_seq_length,
                                                       self.num_attention_heads * self.size_per_head])
        else:
            context_layer = tf.reshape(context_layer, [self.batch_size, self.from_seq_length,
                                                       self.num_attention_heads * self.size_per_head])
        return context_layer


# AlBERT Attention
class ALBERTAttention(tf.keras.layers.Layer):
    def __init__(self,
                 num_attention_heads=1,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 batch_size=None,
                 from_seq_length=None,
                 to_seq_length=None,
                 use_einsum=True,
                 name=None,
                 **kwargs
                 ):
        super(ALBERTAttention, self).__init__(name=name, **kwargs)
        self.num_attention_heads = num_attention_heads
        self.query_act = query_act
        self.key_act = key_act
        self.value_act = value_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.use_einsum = use_einsum

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        size_per_head = int(input_shape[2] / self.num_attention_heads)
        self.q = dense.DenseLayer3d(self.num_attention_heads, size_per_head,
                                    create_initializer(self.initializer_range),
                                    self.query_act,
                                    self.use_einsum, "query")

        self.k = dense.DenseLayer3d(self.num_attention_heads, size_per_head,
                                    create_initializer(self.initializer_range),
                                    self.key_act,
                                    self.use_einsum, "key")

        self.v = dense.DenseLayer3d(self.num_attention_heads, size_per_head,
                                    create_initializer(self.initializer_range), self.value_act,
                                    self.use_einsum, "value")
        self.drop_out = tf.keras.layers.Dropout(self.attention_probs_dropout_prob)
        self.built = True

    def call(self, from_tensor, to_tensor=None, attention_mask=None, is_training=True):
        from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")
        if len(from_shape) == 3:
            self.batch_size = from_shape[0]
            self.from_seq_length = from_shape[1]
            self.to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (self.batch_size is None or self.from_seq_length is None or self.to_seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")
        # `query_layer` = [B, F, N, H]
        q = self.q(from_tensor)
        # `key_layer` = [B, T, N, H]
        k = self.k(to_tensor)
        # `value_layer` = [B, T, N, H]
        v = self.v(to_tensor)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        if attention_mask is not None:
            attention_mask = tf.reshape(
                attention_mask, [self.batch_size, 1, self.to_seq_length, 1])
        # 'new_embeddings = [B, N, F, H]'
        logits = tf.linalg.matmul(q, k, transpose_b=True)
        logits = tf.multiply(logits, 1.0 / math.sqrt(float(get_shape_list(q)[-1])))

        if attention_mask is not None:
            # `attention_mask` = [B, T]
            from_shape = get_shape_list(q)
            if len(from_shape) == 4:
                broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], 1], tf.float32)
            elif len(from_shape) == 5:
                # from_shape = [B, N, Block_num, block_size, depth]#
                broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], from_shape[3],
                                          1], tf.float32)
            attention_mask = tf.matmul(broadcast_ones,
                                       tf.cast(attention_mask, tf.float32), transpose_b=True)
            adder = (1.0 - attention_mask) * -10000.0
            logits += adder

        attention_probs = tf.math.softmax(logits, name="attention_probs")
        attention_probs = self.drop_out(attention_probs, training=is_training)
        context_layer = tf.linalg.matmul(attention_probs, v)

        return tf.transpose(context_layer, [0, 2, 1, 3])


class HieAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, attention_size,
                 w_initializer=None, b_initializer=None,
                 u_initializer=None,
                 **kwargs):
        super(HieAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.W_initializer = tf.keras.initializers.get(w_initializer)
        self.B_initializer = tf.keras.initializers.get(b_initializer)
        self.U_initializer = tf.keras.initializers.get(u_initializer)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=[self.hidden_size, self.attention_size],
            initializer=self.W_initializer,
        )
        self.B = self.add_weight(
            name="B",
            shape=[self.attention_size],
            initializer=self.B_initializer,
        )
        self.U = self.add_weight(
            name="U",
            shape=[self.attention_size],
            initializer=self.U_initializer,
        )

    def call(self, encoder_output):  # [batch,sequence_len,feats_dim]
        if self.hidden_size != encoder_output.shape[-1]:
            raise ValueError("Dim of {} and {} must equal".format("hidden_size", "encode_input"))
        U = tf.math.tanh(tf.tensordot(encoder_output, self.W, axes=1) + self.B)  # [batch,sequence_len, attention_size]
        A = tf.tensordot(U, self.U, axes=1)  # [batch,sequence_len]
        alphas = tf.math.softmax(A)  # [batch,sequence_len]
        output = tf.math.reduce_sum(encoder_output * tf.expand_dims(alphas, -1), 1)  # [batch,sequence_len,feats_dim]
        return output, alphas

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            "attention_size": self.attention_size,
            'v_initializer': tf.keras.initializers.serialize(self.W_initializer),
            'w_initializer': tf.keras.initializers.serialize(self.B_initializer),
            'u_initializer': tf.keras.initializers.serialize(self.U_initializer),
        }
        base_config = super(HieAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# Bahdanau2015
class BahdanauAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size,
                 v_initializer=None,
                 w_initializer=None,
                 u_initializer=None,
                 use_bias=True,
                 b_initializer='zero',
                 name='att', **kwargs):
        super(BahdanauAttentionLayer, self).__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.v_initializer = tf.keras.initializers.get(v_initializer)
        self.w_initializer = tf.keras.initializers.get(w_initializer)
        self.u_initializer = tf.keras.initializers.get(u_initializer)
        self.b_initializer = tf.keras.initializers.get(b_initializer)

    def build(self, input_shape):
        inner_size = input_shape.as_list()
        self.V = self.add_weight(
            name="V",
            shape=[self.hidden_size],
            initializer=self.v_initializer,
        )
        self.W = self.add_weight(
            name="W",
            shape=[inner_size[-1], self.hidden_size],
            initializer=self.w_initializer,
        )
        self.U = self.add_weight(
            name="U",
            shape=[inner_size[-1], self.hidden_size],
            initializer=self.u_initializer,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.hidden_size],
                initializer=self.b_initializer,
            )

        self.attn = tf.keras.layers.Dense(self.hidden_size, activation="tanh")
        self.built = True

    def call(self, hidden_state, encoder_outputs, training=False):
        """
        :param hidden_state: [B,D]
        :param encoder_outputs: [B,T,D]
        :return:
        """
        hidden_inner_dim = hidden_state.get_shape().as_list()[-1]
        encoder_inner_dim = encoder_outputs.get_shape().as_list()[-1]
        if hidden_inner_dim != encoder_inner_dim:
            raise ValueError("The last shape of hidden_state and encoder_outputs must equal!")
        dens1 = tf.tensordot(tf.expand_dims(hidden_state, 1), self.W, axes=(2, 1))
        dens2 = tf.tensordot(encoder_outputs, self.U, axes=(2, 1))
        tanh_ = self.attn(dens1 + dens2 + self.bias) if self.use_bias else self.attn(dens1 + dens2)
        cij = tf.tensordot(tanh_, self.V, axes=(2, 0))
        alphas = tf.keras.backend.softmax(cij)
        output = tf.math.reduce_sum(encoder_outputs * tf.expand_dims(alphas, -1), axis=1)  # [B*T*2H]
        return output, alphas

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            "use_bias": self.use_bias,
            'v_initializer': tf.keras.initializers.serialize(self.v_initializer),
            'w_initializer': tf.keras.initializers.serialize(self.w_initializer),
            'u_initializer': tf.keras.initializers.serialize(self.u_initializer),
            'b_initializer': tf.keras.initializers.serialize(self.b_initializer),
        }
        base_config = super(BahdanauAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# GPT Attention

class GPTAttention(tf.keras.layers.Layer):
    def __init__(self,
                 num_attention_heads=12,
                 initializer_range=0.02,
                 resid_out_rate=0.0,
                 attention_probs_dropout_prob=0.0,
                 scale=True,
                 name=None,
                 **kwargs
                 ):
        super(GPTAttention, self).__init__(name=name, **kwargs)
        self.num_attention_heads = num_attention_heads
        self.scale = scale
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.resid_out_rate = resid_out_rate

    def build(self, input_shape):
        self.size_per_head = int(input_shape[-1] / self.num_attention_heads)
        self.c_att = tf.keras.layers.Dense(
            # 12*64
            self.num_attention_heads * self.size_per_head * 3,
            name="c_attn",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        self.c_proj = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            name="c_proj",
            kernel_initializer=create_initializer(self.initializer_range)
        )

        self.resid_out = tf.keras.layers.Dropout(self.resid_out_rate)

        self.drop_out = tf.keras.layers.Dropout(self.attention_probs_dropout_prob)
        self.built = True

    def causal_attention_mask(self, nd, ns, dtype):
        # 下半角矩阵
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i > j - ns + nd
        return tf.cast(m, dtype)

    def softmax(self, x, axis=-1):
        x = x - tf.reduce_max(x, axis=axis, keepdims=True)
        ex = tf.exp(x)
        return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

    def call(self, from_tensor, layer_past=None, is_training=True):
        """

        :param from_tensor: [B,T,H]
        :param layer_past:
        :param attention_mask:
        :param head_mask:
        :param is_training:
        :return:
        """
        from_shape = get_shape_list(from_tensor, expected_rank=[3])
        self.batch_size = from_shape[0]
        self.from_seq_length = from_shape[1]

        from_tensor = reshape_to_matrix(from_tensor)  # [B*T,Dim]
        output = self.c_att(from_tensor)  # [B*T,3*N*H]
        q, k, v = tf.split(output, 3, axis=1)
        # q, k, v = tf.split(output, 3, axis=2)
        # [B,N,T,H]
        q = transpose_for_scores(q, self.batch_size, self.num_attention_heads,
                                 self.from_seq_length, self.size_per_head)
        k = transpose_for_scores(k, self.batch_size, self.num_attention_heads,
                                 self.from_seq_length, self.size_per_head)
        v = transpose_for_scores(v, self.batch_size, self.num_attention_heads,
                                 self.from_seq_length, self.size_per_head)
        present = tf.stack([k, v], axis=1)
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=1)
            k = tf.concat([past_key, k], axis=-2)
            v = tf.concat([past_value, v], axis=-2)

        # 'new_embeddings = [B, N, T, T]'
        distance = tf.linalg.matmul(q, k, transpose_b=True)
        if self.scale:
            distance = distance*tf.math.rsqrt(float(get_shape_list(v)[-1]))

        _, _, from_length, to_length = get_shape_list(distance)
        distance_b = self.causal_attention_mask(from_length, to_length, dtype=distance.dtype)
        distance_b = tf.reshape(distance_b, [1, 1, from_length, to_length])
        distance = distance * distance_b - 1e10 * (1 - distance_b)

        attention_probs = self.softmax(distance, axis=-1)
        attention_probs = self.drop_out(attention_probs, training=is_training)

        context_layer = tf.linalg.matmul(attention_probs, v)

        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        c_shape = get_shape_list(context_layer)
        # [B,T,N,H] > [B*T,N*H]
        context_layer = tf.reshape(context_layer, [c_shape[0]* c_shape[1]] + [c_shape[-2] * c_shape[-1]])
        output = self.c_proj(context_layer)  #
        output = tf.reshape(output, [c_shape[0], c_shape[1], -1])
        output = self.resid_out(output,training=is_training)

        return output, present
