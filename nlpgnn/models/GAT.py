#! encoding="utf-8"
import tensorflow as tf

from nlpgnn.gnn.GATConv import GraphAttentionConvolution
from nlpgnn.gnn.utils import GNNInput


class GATLayer(tf.keras.Model):
    def __init__(self, hidden_dim=8, num_class=3, heads=8, dropout_rate=0.4, **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.gc1 = GraphAttentionConvolution(hidden_dim,
                                             heads=heads,
                                             concat=True,
                                             dropout_rate=dropout_rate,
                                             name='gat1')
        self.gc2 = GraphAttentionConvolution(num_class,
                                             heads=1,
                                             concat=True,
                                             dropout_rate=dropout_rate,
                                             name='gat2')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, node_embeddings, adjacency_lists, training=True):
        node_embeddings = self.dropout1(node_embeddings, training=training)
        x = self.gc1(GNNInput(node_embeddings, adjacency_lists), training)
        x = tf.nn.elu(x)
        x = self.dropout2(x, training=training)
        x = self.gc2(GNNInput(x, adjacency_lists), training)
        return tf.math.softmax(x, -1)

    def predict(self, node_embeddings, adjacency_lists, training=False):
        return self(node_embeddings, adjacency_lists, training)
