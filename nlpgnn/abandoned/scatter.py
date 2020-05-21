#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf


def broadcast(src, other, dim: int):
    if dim < 0:
        dim = len(other.shape) + dim
    if len(src.shape) == 1:
        for _ in range(0, dim):
            src = tf.expand_dims(src, 0)
    for _ in range(len(src.shape), len(other.shape)):
        src = tf.expand_dims(src, -1)

    # src = src.expand_as(other)
    shape = other.get_shape().as_list()
    shape[0] = 1
    src = tf.tile(src, shape)
    return tf.cast(src, tf.int64)


def scatter_sum(src, index, dim: int = -1,
                out=None,
                dim_size=None):
    index = broadcast(index, src, dim)
    if out is None:
        size = src.get_shape().as_list()
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = tf.zeros(size, dtype=src.dtype)
        print(out)
        print(index)
        print(src)
        return tf.tensor_scatter_nd_add(out, index, src)
    else:
        return tf.tensor_scatter_nd_add(out, index, src)


