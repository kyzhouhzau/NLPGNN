#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf


class BeamSearchDecoder():
    def __init__(self,
                 cell,
                 beam_width,
                 output_layer,
                 length_penalty_weight=0.0):
        self.cell = cell
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.dense = output_layer

    def __call__(self, encode_output, max_length):
        for _ in range(max_length):
            out, h_state = self.cell(encode_output, training=False)
            scores = self.dense(out)
            indices = tf.math.top_k(out, k=3)
            k_state = tf.gather(h_state, indices)
            for encode_output in k_state:
                pass

            encode_output = h_state


class GreedyDecoder():
    def __init__(self, cell, output_layer):
        self.decoder = cell
        self.dense = output_layer

    def __call__(self, encoder_output, init_input, max_length):
        decode_batch = tf.zeros((encoder_output.shape[0], max_length))
        cell_input = init_input
        for i in range(max_length):
            out, encoder_output = self.decoder(cell_input, initial_state=encoder_output)
            scores = self.dense(out)
            indices = tf.math.top_k(scores, k=1)
            decode_batch[:, i] = indices
            cell_input = tf.gather(out, indices)

        return encoder_output


if __name__ == "__main__":
    encode_output = tf.constant([[0.6, 0.4, 0.2, 0.1], [0.2, 0.7, 0.1, 0.2]])
    cell = tf.keras.layers.RNN(10, return_state=True)
    dense = tf.keras.layers.Dense(5, activation="softmax")
    x = GreedyDecoder(cell, dense)
    init_input = tf.constant([[0.2, 0.3, 0.4, 0.1], [0.2, 0.3, 0.4, 0.1]])
    encode_output = x(encode_output, init_input, 3)
    print(encode_output)
