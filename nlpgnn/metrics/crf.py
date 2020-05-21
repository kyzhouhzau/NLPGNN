#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf

"""
序列生成概率记作P(y_1,y_2...,y_n|x_1,x_2,...x_n) = exp(F(y_1,y_2,...y_n|x_1,x_2,...x_n))/Z(X)

取负对数：
-log(P(y_1,y_2...,y_n|x_1,x_2,...x_n))
=-(\sum_{k=0}^{n-1}h(y_{k+1};X) +\sum_{k=1}^{n-1}g(y_k,y_{k+1}))+log(Z(X))

其中函数h是序列每一个t时刻的预测得分，函数g是当前t时刻标签下一个预测为y_{k+1}的概率，两者共同构成
"crf_sequence_score"得分。

Z(X)是所有可能的路径的得分实际上log(Z(X))=\log(\sum \exp a_i), a_i表示第i条可能的路径。
"""


class CrfLogLikelihood(tf.keras.layers.Layer):
    def __init__(self, name='crf', **kwargs):
        super(CrfLogLikelihood, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        num_tags = input_shape[2]
        initializer = tf.keras.initializers.GlorotUniform()
        self.transition_params = self.add_weight(
            shape=[num_tags, num_tags],
            initializer=initializer,
            name="transitions")

    def call(self, inputs, tag_indices, sequence_lengths):
        # cast type to handle different types
        tag_indices = tf.cast(tag_indices, dtype=tf.int32)
        sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
        sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths,
                                             self.transition_params)
        # log_sum(exp(所有可能的路径得分))，对所有路径得分的exp求sum后取log
        log_norm = crf_log_norm(inputs, sequence_lengths, self.transition_params)
        # Normalize the scores to get the log-likelihood per example.
        log_likelihood = sequence_scores - log_norm
        return log_likelihood, self.transition_params

    def crf_decode(selfself, potentials, transition_params, sequence_length):
        sequence_length = tf.cast(sequence_length, dtype=tf.int32)
        initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(initial_state, axis=[1])
        inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])

        sequence_length_less_one = tf.maximum(
            tf.constant(0, dtype=tf.int32), sequence_length - 1)

        backpointers, last_score = crf_decode_forward(
            inputs, initial_state, transition_params, sequence_length_less_one)

        backpointers = tf.reverse_sequence(
            backpointers, sequence_length_less_one, seq_axis=1)

        initial_state = tf.cast(tf.argmax(last_score, axis=1), dtype=tf.int32)
        initial_state = tf.expand_dims(initial_state, axis=-1)

        decode_tags = crf_decode_backward(backpointers, initial_state)
        decode_tags = tf.squeeze(decode_tags, axis=[2])
        decode_tags = tf.concat([initial_state, decode_tags], axis=1)
        decode_tags = tf.reverse_sequence(
            decode_tags, sequence_length, seq_axis=1)

        best_score = tf.reduce_max(last_score, axis=1)
        return decode_tags, best_score


def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
    # trans score
    binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                     transition_params)
    sequence_scores = unary_scores + binary_scores
    return sequence_scores


def crf_unary_score(tag_indices, sequence_lengths, inputs):
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    batch_size = tf.shape(inputs)[0]
    max_seq_len = tf.shape(inputs)[1]
    num_tags = tf.shape(inputs)[2]

    flattened_inputs = tf.reshape(inputs, [-1])

    offsets = tf.expand_dims(tf.range(batch_size) * max_seq_len * num_tags, 1)  # 计算每一个batch起始位置
    offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)  # 计算每一个step的预测结果的起始位置
    # Use int32 or int64 based on tag_indices' dtype.
    if tag_indices.dtype == tf.int64:
        offsets = tf.cast(offsets, tf.int64)
    flattened_tag_indices = tf.reshape(offsets + tag_indices, [-1])  # 这就获得每一个标签所在的index

    unary_scores = tf.reshape(
        tf.gather(flattened_inputs, flattened_tag_indices),
        [batch_size, max_seq_len])  # 将每一个标签所在的index取出来，获得模型预测得分

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32)

    unary_scores = tf.reduce_sum(unary_scores * masks, 1)
    return unary_scores  # 获得模型预测得分


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    num_tags = tf.shape(transition_params)[0]
    num_transitions = tf.shape(tag_indices)[1] - 1

    # Truncate by one on each side of the sequence to get the start and end
    # indices of each transition.
    start_tag_indices = tf.slice(tag_indices, [0, 0], [-1, num_transitions])
    end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

    # Encode the indices in a flattened representation.
    # 太巧妙了
    flattened_transition_indices = start_tag_indices * \
                                   num_tags + end_tag_indices
    flattened_transition_params = tf.reshape(transition_params, [-1])

    # Get the binary scores based on the flattened representation.
    binary_scores = tf.gather(flattened_transition_params,
                              flattened_transition_indices)
    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32)
    truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
    binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)
    return binary_scores


def crf_log_norm(inputs, sequence_lengths, transition_params):
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    # Split up the first and rest of the inputs in preparation for the forward
    # algorithm.
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])
    """Forward computation of alpha values."""
    rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
    # Compute the alpha values in the forward algorithm in order to get the
    # partition function.

    alphas = crf_forward(rest_of_input, first_input, transition_params,
                         sequence_lengths)
    log_norm = tf.reduce_logsumexp(alphas, [1])

    return log_norm


def crf_forward(inputs, state, transition_params, sequence_lengths):
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    last_index = tf.maximum(
        tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1)
    inputs = tf.transpose(inputs, [1, 0, 2])
    transition_params = tf.expand_dims(transition_params, 0)

    def _scan_fn(_state, _inputs):  # 递归函数
        _state = tf.expand_dims(_state, 2)
        # 初始状态往任意点转移的概率,其中每一个初始状态可以向N个其他状态转移。、
        # 对于t时刻每一个特定的batch中向量中的每一个可能的点下一时刻都有N种可能的转移结果
        # 这里通过[B,N,1]+[1,N,N]的相加实现
        transition_scores = _state + transition_params
        new_alphas = _inputs + tf.reduce_logsumexp(transition_scores, [1])
        return new_alphas

    all_alphas = tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])
    # add first state for sequences of length 1
    all_alphas = tf.concat([tf.expand_dims(state, 1), all_alphas], 1)

    idxs = tf.stack([tf.range(tf.shape(last_index)[0]), last_index], axis=1)
    return tf.gather_nd(all_alphas, idxs)


def crf_decode_forward(inputs, state, transition_params, sequence_lengths):
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    mask = tf.sequence_mask(sequence_lengths, tf.shape(inputs)[1])
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
    crf_fwd_layer = tf.keras.layers.RNN(
        crf_fwd_cell, return_sequences=True, return_state=True)
    return crf_fwd_layer(inputs, state, mask=mask)


class CrfDecodeForwardRnnCell(tf.keras.layers.AbstractRNNCell):
    """Computes the forward decoding in a linear-chain CRF."""

    def __init__(self, transition_params, **kwargs):
        """Initialize the CrfDecodeForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. This matrix is expanded into a
            [1, num_tags, num_tags] in preparation for the broadcast
            summation occurring within the cell.
        """
        super(CrfDecodeForwardRnnCell, self).__init__(**kwargs)
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.shape[0]

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def build(self, input_shape):
        super(CrfDecodeForwardRnnCell, self).build(input_shape)

    def call(self, inputs, state):
        """Build the CrfDecodeForwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous step's
                score values.
        Returns:
          backpointers: A [batch_size, num_tags] matrix of backpointers.
          new_state: A [batch_size, num_tags] matrix of new score values.
        """
        state = tf.expand_dims(state[0], 2)
        transition_scores = state + self._transition_params
        new_state = inputs + tf.reduce_max(transition_scores, [1])
        backpointers = tf.argmax(transition_scores, 1)
        backpointers = tf.cast(backpointers, dtype=tf.int32)
        return backpointers, new_state


def crf_decode_backward(inputs, state):
    """Computes backward decoding in a linear-chain CRF.
    Args:
      inputs: A [batch_size, num_tags] matrix of
            backpointer of next step (in time order).
      state: A [batch_size, 1] matrix of tag index of next step.
    Returns:
      new_tags: A [batch_size, num_tags]
        tensor containing the new tag indices.
    """
    inputs = tf.transpose(inputs, [1, 0, 2])

    def _scan_fn(state, inputs):
        state = tf.squeeze(state, axis=[1])
        idxs = tf.stack([tf.range(tf.shape(inputs)[0]), state], axis=1)
        new_tags = tf.expand_dims(tf.gather_nd(inputs, idxs), axis=-1)
        return new_tags

    return tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])
