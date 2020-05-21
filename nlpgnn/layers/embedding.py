#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
@Tensorflow 2.0
All of the following Code was follow Google BERT!
"""
import tensorflow as tf
from nlpgnn.tools import create_initializer, get_shape_list


# for bert
class WDEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 initializer_range=0.02,
                 word_embedding_name="word_embeddings",
                 use_one_hot_embedding=False,
                 name=None,
                 **kwargs):
        super(WDEmbedding, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
        self.word_embedding_name = word_embedding_name
        self.use_one_hot_embedding = use_one_hot_embedding

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        self.embedding_table = self.add_weight(
            name=self.word_embedding_name,
            dtype=tf.keras.backend.floatx(),
            shape=[self.vocab_size, self.embedding_size],
            initializer=create_initializer(self.initializer_range),
            trainable=True,
        )
        self.built = True

    def call(self, input_ids):
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])
        flat_input_ids = tf.reshape(input_ids, [-1])
        if self.use_one_hot_embedding:
            one_hot_input_ids = tf.keras.backend.one_hot(flat_input_ids, self.vocab_size)
            output = tf.linalg.matmul(one_hot_input_ids, self.embedding_table)
        else:
            output = tf.gather(self.embedding_table, flat_input_ids)
        input_shape = get_shape_list(input_ids)
        output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        return output


# for bert
class SegPosEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 use_token_type=True,
                 hidden_dropout_prob=0.01,
                 token_type_vocab_size=2,
                 token_type_embedding_name="token_type_embeddings",
                 use_position_embeddings=True,
                 position_embedding_name="position_embeddings",
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 use_one_hot_embedding=True,
                 name=None,
                 **kwargs):

        super(SegPosEmbedding, self).__init__(name=name, **kwargs)
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.token_type_embedding_name = token_type_embedding_name
        self.use_position_embeddings = use_position_embeddings
        self.position_embedding_name = position_embedding_name
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_one_hot_embedding = use_one_hot_embedding

    def build(self, input_shape):

        input_ids_shape = input_shape
        self.input_spec = tf.keras.layers.InputSpec(shape=input_ids_shape)

        self.token_type_table = self.add_weight(
            name=self.token_type_embedding_name,
            shape=[self.token_type_vocab_size, input_shape[2]],
            dtype=tf.keras.backend.floatx(),
            initializer=create_initializer(self.initializer_range),
            trainable=True
        )

        self.full_position_embeddings = self.add_weight(
            name=self.position_embedding_name,
            shape=[self.max_position_embeddings, input_shape[2]],
            dtype=tf.keras.backend.floatx(),
            initializer=create_initializer(self.initializer_range),
            trainable=True)

        self.drop_out = tf.keras.layers.Dropout(self.hidden_dropout_prob)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="LayerNorm")

        self.built = True

    def call(self, input_tensor, token_type_ids=None, is_training=True):
        inputshape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = inputshape[0]
        seq_length = inputshape[1]
        width = inputshape[2]
        output = input_tensor
        # segment features
        if self.use_token_type:
            if token_type_ids is None:
                raise ValueError("token_type_ids must be specified if use_token_type is True")
            if self.use_one_hot_embedding:
                flat_token_type_ids = tf.reshape(token_type_ids, [-1])
                one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
                token_type_embeddings = tf.linalg.matmul(one_hot_ids, self.token_type_table)
                token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
            else:
                token_type_embeddings = tf.gather(self.token_type_table, token_type_ids)
            output += token_type_embeddings
        # position features
        if self.use_position_embeddings:
            position_embeddings = tf.slice(self.full_position_embeddings, [0, 0], [seq_length, -1])
            # num_dims = len(output.shape.as_list())
            num_dims = len(output.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings
        output = self.layer_norm(output)
        # in official work they not use training
        output = self.drop_out(output, training=is_training)
        return output


# TODO
class ZengPosEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ZengPosEmbedding, self).__init__(**kwargs)
        pass

    def call(self):
        pass


# for bert
class WTEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 initializer_range=0.02,
                 word_embedding_name="word_embeddings",
                 name=None,
                 **kwargs):
        super(WTEmbedding, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
        self.word_embedding_name = word_embedding_name

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        self.embedding_table = self.add_weight(
            name=self.word_embedding_name,
            dtype=tf.keras.backend.floatx(),
            shape=[self.vocab_size, self.embedding_size],
            initializer=create_initializer(self.initializer_range),
            trainable=True,
        )
        self.built = True

    def call(self, input_ids):
        output = tf.gather(self.embedding_table, input_ids)
        return output
