#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from nlpgnn.tools import get_shape_list, create_initializer
from nlpgnn.layers.embedding import WTEmbedding
from nlpgnn.layers.gpt2_transformer import GPT2Transformer
from nlpgnn.layers import normalization


class GPT2(tf.keras.layers.Layer):
    def __init__(self,
                 param=None,
                 vocab_size=30000,  # vocab_size
                 hidden_size=768,  # n_embed
                 max_position_length=1024,  # n_ctx = 1024
                 embedding_drop_rate=0.0,  # embd_pdrop
                 layer_norm_epsilon=1e-5,  # layer_norm_epsilon
                 num_attention_heads=12,  # n_head = 12
                 initializer_range=0.02,  # initializer_range
                 num_hidden_layers=12,  # n_layer
                 batch_size=10,
                 attention_probs_dropout_prob=0.0,  # attn_pdrop
                 resid_out_rate=0.0,
                 name="gpt2",
                 **kwargs):
        super(GPT2, self).__init__(name=name, **kwargs)
        self.vocab_size = param.get("n_vocab", vocab_size)
        self.batch_size = param.get("batch_size", batch_size)
        self.num_attention_heads = param.get("n_head", num_attention_heads)
        self.num_hidden_layers = param.get("n_layer", num_hidden_layers)
        self.hidden_size = param.get("n_embd", hidden_size)
        self.max_position_length = param.get("n_ctx", max_position_length)
        self.resid_out_rate = param.get("resid_out_rate", resid_out_rate)

        self.initializer_range = param.get("initializer_range", initializer_range)
        self.attention_probs_dropout_prob = param.get("attention_probs_dropout_prob", attention_probs_dropout_prob)
        self.embedding_drop_rate = embedding_drop_rate
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        self.token_embedding = WTEmbedding(vocab_size=self.vocab_size,
                                           embedding_size=self.hidden_size,
                                           initializer_range=self.initializer_range,
                                           word_embedding_name="embedding",
                                           name="wte")
        # position embedding
        self.posembedding = tf.keras.layers.Embedding(
            self.max_position_length,
            self.hidden_size,
            embeddings_initializer=create_initializer(self.initializer_range),
            name="wpe",
        )
        self.embedding_drop = tf.keras.layers.Dropout(self.embedding_drop_rate)

        self.encoder_layers = []
        for layer_idx in range(self.num_hidden_layers):
            self.encoder_layer = GPT2Transformer(
                num_attention_heads=self.num_attention_heads,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                initializer_range=self.initializer_range,
                epsilon=self.layer_norm_epsilon,
                resid_out_rate=self.resid_out_rate,
                name="h{}".format(layer_idx)
            )
            self.encoder_layers.append(self.encoder_layer)

        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon, name='ln_f')

        # self.ln_f = normalization.GPTNorm(epsilon=self.layer_norm_epsilon, name='ln_f')

        self.built = True

    def past_shape(self):
        return [self.batch_size, self.num_hidden_layers, 2,
                self.num_attention_heads, None,
                self.hidden_size // self.num_attention_heads]

    def position_ids(self, tokens, past_length):
        batch_size = tf.shape(tokens)[0]
        nsteps = tf.shape(tokens)[1]
        return self.expand_tile(past_length + tf.range(nsteps), batch_size)

    def expand_tile(self, value, size):
        value = tf.convert_to_tensor(value, name='value')
        ndims = value.shape.ndims
        return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)

    def call(self, input_ids, past=None, training=False):
        results = {}
        input_ids = tf.cast(input_ids, tf.int32)
        batch_size = get_shape_list(input_ids)[0]
        past_length = 0 if past is None else get_shape_list(past)[-2]
        inputs_embeds = self.token_embedding(input_ids)
        position_embeds = self.posembedding(self.position_ids(input_ids, past_length))

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.embedding_drop(hidden_states, training=training)

        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.num_hidden_layers

        for i, (block, layer_past) in enumerate(zip(self.encoder_layers, pasts)):
            hidden_states, present = block(hidden_states, layer_past, training)
            presents.append(present)
        results['presents'] = tf.stack(presents, axis=1)
        output = self.ln_f(hidden_states)

        output_flat = tf.reshape(output, [-1, self.hidden_size])
        logits = tf.matmul(output_flat, self.token_embedding.embedding_table, transpose_b=True)
        logits = tf.reshape(logits, [batch_size, -1, self.vocab_size])
        results['logits'] = logits
        return results
