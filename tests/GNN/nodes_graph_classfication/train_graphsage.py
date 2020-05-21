#! encoding:utf-8
import tensorflow as tf
import numpy as np
import time
import sys
from tqdm import tqdm
from nlpgnn.datas import TUDataset
from nlpgnn.metrics import Losess, Metric
from nlpgnn.models import GraphSAGE
from nlpgnn.gnn.utils import merge_batch_graph
from sklearn.metrics import classification_report

dim = 100
num_class = 2
drop_rate = 0.5
epoch = 200

lr = 0.001
split = 10  # 10-fold

accs_all = []
dataloader = TUDataset(name="MUTAG", split=split)

for block_index in range(split):

    model = GraphSAGE(dim, num_class, drop_rate)

    optimize = tf.optimizers.Adam(lr)

    cross_entropy = Losess.MaskSparseCategoricalCrossentropy()
    acc_score = Metric.SparseAccuracy()

    train_data, test_data = dataloader.load(block_index=block_index)
    train_bar = tqdm(range(epoch), unit="epoch", file=sys.stdout)
    for i in train_bar:
        t = time.time()
        label_train = []
        predict_train = []
        loss_train = []
        for x, y, edge_index, edge_attr, batch in dataloader.sample(train_data,iterator_per_epoch=6,

                                                                    batch_size=32, mode="train"):
            x, y, edge_index, edge_attr, batch = merge_batch_graph(x, y, edge_index,
                                                                   edge_attr, batch)
            x = tf.cast(x, tf.float32)
            with tf.GradientTape() as tape:
                predict = model(x, edge_index, batch, training=True)
                loss = cross_entropy(y, predict)
                predict_train.extend(tf.argmax(predict, -1))
                label_train.extend(y)
                loss_train.append(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimize.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        report = classification_report(label_train, predict_train, digits=4, output_dict=True)
        acc = report["accuracy"]
        train_bar.set_description('loss: {:.4f}, acc: {:.0f}%'.format(np.mean(loss_train), 100. * acc))
        if i != 0 and i % 50 == 0:
            lr_rate = optimize._get_hyper("learning_rate")
            optimize._set_hyper('learning_rate', lr_rate * 0.5)
    # ------------------------------------------------------------------------
    label_test = []
    predict_test = []
    for x, y, edge_index, edge_attr, batch in dataloader.sample(test_data,iterator_per_epoch=6,
                                                                batch_size=32, mode="test"):
        x, y, edge_index, edge_attr, batch = merge_batch_graph(x, y, edge_index,
                                                               edge_attr, batch)
        x = tf.cast(x, tf.float32)
        t_predict = model.predict(x, edge_index, batch, training=False)
        predict_test.extend(tf.argmax(t_predict, -1))
        label_test.extend(y)
    report = classification_report(label_test, predict_test, digits=4, output_dict=True)
    acc = report["accuracy"]
    accs_all.append(acc)
    print("Test: Acc {:.5f}".format(acc))

print("ACC: {:.4f}±{:.4f}".format(np.mean(accs_all), np.std(accs_all)))
