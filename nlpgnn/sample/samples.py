#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf


def top_k_logits(logits, k):
    if k == 0:
        return logits
    else:
        values, indices = tf.math.top_k(logits, k=k)
        min_values = tf.expand_dims(values[:, -1], -1)
        # 这个用法好，可以按元素判断并做修改
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def top_p_logits(logits, p):
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    # cumsum 累加
    cumulative_probs = tf.math.cumsum(tf.math.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    min_values = tf.expand_dims(min_values,-1)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(model, param, length=None,
                    start_token=None,
                    batch_size=None,
                    context=None,
                    temperature=1,
                    top_k=0, top_p=1
                    ):
    if length is None:
        length = 1024 // 2
    batch_size = param.get("batch_size", batch_size)

    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)
        # context = tf.tile(tf.constant([start_token]), [batch_size, 1])

    past_shape = [batch_size, param.get("n_layer"), 2,
                  param.get("n_head"), -1,
                  param.get("n_embd") // param.get("n_head")]
    past_shape_none = [batch_size, param.get("n_layer"), 2,
                       param.get("n_head"), None,
                       param.get("n_embd") // param.get("n_head")]

    def step(tokens, past=None):
        lm_output = model.predict(tokens, past)
        logits = lm_output["logits"][:, :, :param.get("n_vocab")]
        presents = lm_output['presents']
        presents = tf.reshape(presents, past_shape)
        return {
            'logits': logits,
            'presents': presents,
        }
    with tf.keras.backend.name_scope('sample_sequence'):

        def body(past, prev, output):
            next_outputs = step(prev, past)
            logits = next_outputs['logits'][:, -1, :] / tf.cast(temperature, tf.float32)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)

            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]

        past, prev, output = body(None, context, context)

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf.TensorShape(past_shape_none),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )
        return tokens





if __name__ == "__main__":
    logits = tf.constant([[0.1, 0.4, 0.5, 0.0001], [0.2, 0.1, 0.4, 0.8], [0.2, 0.01, 0.4, 8]])

    x = top_p_logits(logits, 0.8)
    print(x)
