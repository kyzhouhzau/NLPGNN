#! encoding="utf-8"
import tensorflow as tf

from nlpgnn.gnn.GCNConv import GraphConvolution
from nlpgnn.gnn.utils import GNNInput


class GCNLayer(tf.keras.Model):
    def __init__(self, hidden_dim, num_class, dropout_rate=0.5, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.gc1 = GraphConvolution(hidden_dim, name='gcn1')
        self.gc2 = GraphConvolution(num_class, name='gcn2')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, node_embeddings, adjacency_lists, training=True):
        node_embeddings = self.dropout1(node_embeddings, training=training)
        x = self.gc1(GNNInput(node_embeddings, adjacency_lists))
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)

        x = self.gc2(GNNInput(x, adjacency_lists))
        return tf.math.softmax(x, -1)

    def predict(self, node_embeddings, adjacency_lists, training=False):
        return self(node_embeddings, adjacency_lists, training)
