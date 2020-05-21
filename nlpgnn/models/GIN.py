#! encoding="utf-8"

from nlpgnn.gnn.GINConv import GINConvolution
from nlpgnn.gnn.utils import *
import tensorflow as tf

Dense = tf.keras.layers.Dense
Drop = tf.keras.layers.Dropout
BatchNorm = tf.keras.layers.BatchNormalization


class ApplyNodeFunc(tf.keras.layers.Layer):
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = BatchNorm()

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = tf.nn.relu(h)
        return h


class MLP(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim, num_layers, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.Denses = []
        for i in range(num_layers - 1):
            self.Denses.append(Dense(hidden_dim))
        self.Denses.append(Dense(output_dim))
        self.batch_norms = []
        for layer in range(num_layers - 1):
            self.batch_norms.append(BatchNorm)

    def call(self, x, training=True):
        h = x
        for i in range(self.num_layers - 1):
            h = tf.nn.relu(self.batch_norms[i](self.Denses[i](h), training=training))
        return self.Denses[-1](h)


class GINLayer(tf.keras.Model):
    def __init__(self, dim=32, num_classes=2, drop_rate=0.5, mlp_layer = 2, layer_num=5, **kwargs):
        super(GINLayer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.norms = []
        self.ginlayers = []
        for i in range(layer_num - 1):
            mlp = MLP(dim, dim, mlp_layer)
            self.ginlayers.append(GINConvolution(ApplyNodeFunc(mlp), train_eps=False))
            self.norms.append(BatchNorm())

        self.denses = []
        self.drops = []
        for i in range(layer_num):
            self.denses.append(Dense(num_classes))
            self.drops.append(tf.keras.layers.Dropout(drop_rate))

    def call(self, node_embeddings, edge_index, batch, training=True):
        edge_index = [edge_index]
        hidden_rep = [node_embeddings]
        for i in range(self.layer_num - 1):
            node_embeddings = self.ginlayers[i](GNNInput(node_embeddings, edge_index), training)
            node_embeddings = self.norms[i](node_embeddings, training=training)
            node_embeddings = tf.nn.relu(node_embeddings)
            hidden_rep.append(node_embeddings)
        score = 0
        for i, h in enumerate(hidden_rep):
            pool_out = batch_read_out(h, batch)
            score += self.drops[i](self.denses[i](pool_out),training=training)

        return tf.math.softmax(score, -1)

    def predict(self, node_embeddings, edge_index, batch, training=False):
        return self(node_embeddings, edge_index, batch, training)
