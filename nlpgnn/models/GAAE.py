#! encoding="utf-8"
import tensorflow as tf

from nlpgnn.gnn.GAAEConv import GraphAttentionAutoEncoder
from nlpgnn.gnn.utils import GNNInput


class GAAELayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=16, num_layers=2, heads=1, **kwargs):
        super(GAAELayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.weight = []
        self.weight.append(self.add_weight(
            shape=(input_dim, self.heads * self.hidden_dim),
            name='wt',
        ))
        for i in range(self.num_layers - 1):
            self.weight.append(self.add_weight(
                shape=(self.hidden_dim, self.heads * self.hidden_dim),
                name='wt',
            ))
        self.encoder_layers = []
        self.decoder_layers = []
        for layer in range(self.num_layers - 1):
            self.encoder_layers.append(GraphAttentionAutoEncoder(self.hidden_dim, heads=self.heads))
            self.decoder_layers.append(GraphAttentionAutoEncoder(self.hidden_dim, heads=self.heads))
        self.encoder_layers.append(GraphAttentionAutoEncoder(self.hidden_dim, heads=self.heads))
        self.decoder_layers.append(GraphAttentionAutoEncoder(input_dim, heads=self.heads))

    def encoder(self, node_embeddings, adjacency_lists, training):
        for layer in range(self.num_layers):
            node_embeddings = self.encoder_layers[layer](GNNInput(node_embeddings, adjacency_lists), self.weight[layer],
                                                         False, training)
        return node_embeddings

    def decoder(self, hidden_embeddings, adjacency_lists, training):
        for layer in range(self.num_layers):
            hidden_embeddings = self.decoder_layers[layer](GNNInput(hidden_embeddings, adjacency_lists),
                                                           self.weight[-(layer+1)],
                                                           True,
                                                           training)
        return hidden_embeddings

    def call(self, node_embeddings, adjacency_lists, training=True):
        hidden_embeddings = self.encoder(node_embeddings, adjacency_lists, training)
        reconstruct_embedding = self.decoder(hidden_embeddings, adjacency_lists, training)
        return hidden_embeddings, reconstruct_embedding

    def predict(self, node_embeddings, adjacency_lists, training=False):
        return self(node_embeddings, adjacency_lists, training)
