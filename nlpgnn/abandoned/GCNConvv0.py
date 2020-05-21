#! encoding="utf-8"
import tensorflow as tf


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self,
                 out_features,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 use_bias=True,
                 **kwargs):

        super(GraphConvolution, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.out_features = out_features
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        in_features = input_shape[1]
        self.weight = self.add_weight(
            shape=(in_features, self.out_features),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel',
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_features,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                name='bias',
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.build = True

    def call(self, input, adj):
        # input:csr_matrix
        support = tf.linalg.matmul(input, self.weight)
        # SparseTensor format expected by sparse_tensor_dense_matmul: sp_a (indices, values)
        output = tf.sparse.sparse_dense_matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def get_config(self):
        config = {'units': self.out_features,
                  'use_bias': self.use_bias,

                  'activation': tf.keras.activations.serialize(self.activation),

                  'kernel_initializer': tf.keras.initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': tf.keras.initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': tf.keras.regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': tf.keras.regularizers.serialize(
                      self.bias_regularizer),
                  'kernel_constraint': tf.keras.constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': tf.keras.constraints.serialize(
                      self.bias_constraint)
                  }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
