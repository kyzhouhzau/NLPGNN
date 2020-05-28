#! encoding:utf-8
import time
import numpy as np
import tensorflow as tf
from nlpgnn.datas import Planetoid
from nlpgnn.metrics import Losess, Metric
from nlpgnn.models import GCNLayer
from nlpgnn.callbacks import EarlyStopping

tf.random.set_seed(10)

hidden_dim = 16
num_class = 6
drop_rate = 0.5
epoch = 200
early_stopping = 10
penalty = 5e-4

# cora, pubmed, citeseer
data = Planetoid(name="citeseer", loop=True, norm=True)

features, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = data.load()



model = GCNLayer(hidden_dim, num_class, drop_rate)

optimizer = tf.keras.optimizers.Adam(0.01)
crossentropy = Losess.MaskCategoricalCrossentropy()
accscore = Metric.MaskAccuracy()
stop_monitor = EarlyStopping(monitor="loss", patience=early_stopping)

# ---------------------------------------------------------
# For train
for p in range(epoch):
    t = time.time()
    with tf.GradientTape() as tape:
        predict = model(features, adj, training=True)
        loss = crossentropy(y_train, predict, train_mask)
        loss += penalty * tf.nn.l2_loss(model.variables[0])

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    predict_v = model.predict(features, adj)
    loss_v = crossentropy(y_val, predict_v, val_mask)
    acc = accscore(y_val, predict_v, val_mask)
    print("Epoch {} | Loss {:.4f} | Acc {:.4f} | Time {:.4f}".format(p, loss_v.numpy(), acc, time.time() - t))
    if stop_monitor(loss_v, model):
        break
# --------------------------------------------------------------------------------------
# For test
predict_t = model.predict(features, adj)
acc = accscore(y_test, predict_t, test_mask)
loss = crossentropy(y_test, predict_t, test_mask)
print("Test Loss {:.4f} | ACC {:.4f}".format(loss.numpy(), acc))
