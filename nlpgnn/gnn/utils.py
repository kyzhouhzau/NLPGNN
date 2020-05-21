#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
from typing import NamedTuple, List
import numpy as np
import tensorflow as tf
from scipy import sparse
import scipy.sparse as sp
from collections import Counter


class GNNInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: List


def add_remain_self_loop(adjacency_lists, num_nodes):
    loop_index = tf.range(0, num_nodes)
    loop_index = tf.expand_dims(loop_index, 1)
    loop_index = tf.tile(loop_index, [1, 2])
    row = adjacency_lists[:, 0]
    col = adjacency_lists[:, 1]
    mask = row != col
    loop_index = tf.concat([adjacency_lists[mask], loop_index], 0)
    return loop_index


def add_self_loop(adjacency_lists, num_nodes):
    loop_index = tf.range(0, num_nodes)
    loop_index = tf.expand_dims(loop_index, 1)
    loop_index = tf.tile(loop_index, [1, 2])
    loop_index = tf.concat([adjacency_lists, loop_index], 0)
    return loop_index


def remove_self_loop(adjacency_lists, edge_attr=None):
    row = adjacency_lists[:, 0]
    col = adjacency_lists[:, 1]
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    adjacency_lists = adjacency_lists[mask]
    return adjacency_lists, edge_attr


def maybe_num_nodes(index, num_nodes):
    return tf.reduce_max(index) + 1 if num_nodes is None else num_nodes


def masksoftmax(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index, num_nodes)
    inter = tf.math.unsorted_segment_max(data=src,
                                         segment_ids=index,
                                         num_segments=num_nodes)
    # out = src - tf.gather(inter, index)# 每一个维度减去最大的特征
    out = src
    out = tf.math.exp(out)
    inter = tf.math.unsorted_segment_sum(data=out, segment_ids=index, num_segments=num_nodes)
    out = out / (tf.gather(inter, index) + 1e-16)
    return out


def coalesce(index, value, n, sort=True):
    """
    index = [[row,col],[row,col],[row,col]]
    value = [v,v,v]
    n: num_nodes
    """
    # if sort:
    # index = index[np.argsort(index[:, 0])]
    # index = np.array(sorted(index.tolist()))
    row = index[:, 0]
    col = index[:, 1]
    idx = np.full((col.size + 1,), -1, dtype=col.dtype)
    idx[1:] = n * row + col
    mask = idx[1:] > idx[:-1]
    if mask.all():
        return index, value
    row = row[mask]
    col = col[mask]
    if value is not None:
        ptr = mask.nonzero()[0]
        ptr = np.concatenate([ptr, np.full((1,), value.size)])
        ptr = rebuilt(ptr)
        value = tf.math.segment_sum(value, ptr)[mask]
        value = value[0] if isinstance(value, tuple) else value
    edge_index = np.array(list(zip(row, col)))
    return edge_index, value


def rebuilt(ptr):
    length = ptr[-1]
    index = ptr[:-1]
    nptr = np.zeros(length, dtype=np.int32)
    nptr[index] = index
    for i in range(length):
        if nptr[i] != i:
            nptr[i] = nptr[i - 1]
    return nptr


def merge_graph(ds, batch=None, name=None):
    window = list(ds.as_numpy_iterator())
    if name == "index":
        assert batch != None
        batch = list(batch.as_numpy_iterator())
        strips = [0] + [i.size for i in batch]
        window = [w + strips[i] for i, w in enumerate(window)]

    if isinstance(window[0], np.int32):
        return tf.concat([window], 0)
    else:
        return tf.concat(window, 0)


def merge_batch_graph(x, y, edge_index, edge_attr, batch):
    if x != None:
        lis_x = [np.array(item) for item in x]
        x = tf.concat(lis_x, 0)
    if y != None:
        lis_y = [np.array(item) for item in y]
        y = tf.concat([lis_y], 0)

    if edge_index != None and batch != None:
        lis_edge_index = [np.array(item) for item in edge_index]
        lis_batch = [np.array(item) for item in batch]
        strips = [0] + [i.size for i in lis_batch]
        strips = np.cumsum(strips)
        nlis_edge_index = [w + strips[i] for i, w in enumerate(lis_edge_index)]
        edge_index = tf.concat(nlis_edge_index, 0)
        batch = tf.concat(lis_batch, 0)

    if edge_attr != None:
        lis_edge_attr = [np.array(item) for item in edge_attr]
        if np.isnan(lis_edge_attr[0]).all():
            edge_attr = None
        else:
            edge_attr = tf.concat(lis_edge_attr, 0)
    # if node != None:
    #     lis_node = [np.array(item) for item in node]
    #     if np.isnan(lis_node[0]).all():
    #         node = None
    #     else:
    #         node = tf.concat(lis_node, 0)
    #     return x, y, edge_index, edge_attr, batch, node

    return x, y, edge_index, edge_attr, batch


def batch_read_out(x, batch, method="sum", size=None):
    n_batch = tf.unique(batch)[1]
    size = tf.reduce_max(n_batch) + 1 if size is None else size
    if method == "sum":
        readout = tf.math.unsorted_segment_sum(x, n_batch, size)
    elif method == "max":
        readout = tf.math.unsorted_segment_max(x, n_batch, size)
    elif method == 'mean':
        readout = tf.math.unsorted_segment_mean(x, n_batch, size)
    return readout


from multiprocessing.dummy import Pool as ThreadPool

pool = ThreadPool(10)


def features2embedding(feature, word2embedding):
    def func_wrap(word2emb):
        def func(item):
            emb = word2emb.get(item.numpy())
            # return tf.cast(emb, dtype=tf.float32)
            return emb

        return func

    # nfeatures = tf.map_fn(func_wrap(word2embedding), tf.cast(feature, tf.float32),parallel_iterations=8)
    nfeatures = list(pool.map(func_wrap(word2embedding), tf.cast(feature, tf.float32)))

    return tf.constant(nfeatures, dtype=tf.float32)
