#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf
import time
import numpy as np
from nlpgnn.datas import Sminarog
from nlpgnn.metrics import Losess, Metric
from nlpgnn.models import TextGCN2019
from nlpgnn.gnn.utils import merge_batch_graph
from nlpgnn.callbacks import EarlyStopping
from sklearn.metrics import classification_report

dim = 100
glove_dim = 100
num_class = 52
drop_rate = 0.5
epoch = 150
penalty = 1e-4
lr = 1e-3

# R8,R52
data = Sminarog(data="R52", data_dir="data", embedding="glove{}".format(glove_dim))
nodes, adjs, edge_attrs, labels, batchs, edge2index, node2index = data.build_graph(mode="train", p=3, k=1)
nodes1, adjs1, edge_attrs1, labels1, batchs1, _, _ = data.build_graph(edge2index, node2index, mode="test", p=3)

def init_weight(index2node, node2embedding):
    weight = []
    for i in range(len(index2node)):
        w = index2node[i]
        if w not in node2embedding:
            w = "<UNK>"
        weight.append(node2embedding.get(w))
    return np.array(weight, np.float32)


class TextGCNDynamicWeight(tf.keras.layers.Layer):
    def __init__(self, dim, num_class, drop_rate, **kwargs):
        super(TextGCNDynamicWeight, self).__init__(**kwargs)
        self.model = TextGCN2019(dim, num_class, drop_rate)
        index2node = {index: node for node, index in node2index.items()}
        self.edge_num = len(set(edge2index.values()))
        self.node_num = len(node2index)
        self.init = init_weight(index2node, data.word2embedding)

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(self.node_num, glove_dim),
            initializer=tf.constant_initializer(self.init),
            trainable=True,
            name="embedding"
        )
        self.ean = self.add_weight(
            shape=(self.edge_num,),
            # regularizer=tf.keras.regularizers.l2(penalty),
            name='ean',
        )
        self.etans = self.add_weight(
            shape=(self.node_num,),
            # regularizer=tf.keras.regularizers.l2(penalty),
            name='etans',

        )

    def call(self, node, adj, edge_attr, batch, training=True):
        edge_attr = tf.cast(edge_attr, dtype=tf.int32)
        edge_weight = tf.gather(self.ean, edge_attr)# 引入预训练
        etans = tf.reshape(tf.gather(self.etans, node), [-1, 1])
        feature = tf.nn.embedding_lookup(self.embedding, node)
        predict = self.model(feature, etans, adj, batch, edge_weight, training=training)
        return predict

    def predict(self, nodes, adj, edge_attr, batch, training=False):
        return self(nodes, adj, edge_attr, batch, training)


accs_all = []
for i in range(10):
    model = TextGCNDynamicWeight(dim, num_class, drop_rate)
    optimize = tf.optimizers.Adam(lr)

    cross_entropy = Losess.MaskSparseCategoricalCrossentropy()
    acc_score = Metric.SparseAccuracy()

    stop_monitor = EarlyStopping(monitor="loss", patience=10, restore_best_weights=False)
    for i in range(epoch):
        loss_train = []
        acc_train = []
        t = time.time()

        for node, label, adj, edge_attr, batch in data.load(nodes[:-500],
                                                            adjs[:-500], labels[:-500],
                                                            edge_attrs[:-500], batchs[:-500],
                                                            batch_size=32):
            node, label, adj, edge_attr, batch = merge_batch_graph(node, label, adj, edge_attr, batch)

            with tf.GradientTape() as tape:
                predict = model(node, adj, edge_attr, batch, training=True)
                loss = cross_entropy(label, predict)
                loss += tf.add_n([tf.nn.l2_loss(v) for v in model.variables
                                  if "embed" and "bias" not in v.name]) * penalty
            grads = tape.gradient(loss, model.trainable_variables)
            optimize.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

        t_loss=t_acc=0
        for node, label, adj, edge_attr, batch in data.load(nodes[-500:], adjs[-500:],
                                                            labels[-500:], edge_attrs[-500:],
                                                            batchs[-500:],
                                                            batch_size=500):
            node, label, adj, edge_attr, batch = merge_batch_graph(node, label, adj, edge_attr, batch)
            t_predict = model.predict(node, adj, edge_attr, batch, training=False)
            t_loss = cross_entropy(label, t_predict)
            t_acc = acc_score(label, t_predict)
        # if i != 0 and i % 10 == 0:
        #     lr_rate = optimize._get_hyper("learning_rate")
        #     optimize._set_hyper('learning_rate', lr_rate * 0.5)
        print("Valid: Epoch {} | Loss {:.4f} | Acc {:.4f} | Time {:.4f}".format(i, t_loss, t_acc,
                                                                                time.time() - t))
        if stop_monitor(current=np.mean(t_loss), model=model):
            break

        pres = []
        labs = []
        for node, label, adj, edge_attr, batch in data.load(nodes1, adjs1, labels1, edge_attrs1,
                                                            batchs1, batch_size=32):
            node, label, adj, edge_attr, batch = merge_batch_graph(node, label, adj,
                                                                   edge_attr, batch)
            # feature = features2embedding(feature, data.word2embedding)
            t_predict = model.predict(node, adj, edge_attr, batch, training=False)

            pres.extend(tf.argmax(t_predict, -1))
            labs.extend(label)
        report = classification_report(labs, pres, digits=4, output_dict=True)
        acc = report["accuracy"]
        print("Test: Acc {:.5f}".format(acc))
        accs_all.append(acc)
print("ACC: {:.5f}±{:.5f}".format(np.mean(accs_all), np.std(accs_all)))
