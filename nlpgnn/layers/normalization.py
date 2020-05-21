#! encoding:utf-8
import tensorflow as tf

constraints = tf.keras.constraints
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers
math_ops = tf.math


class FPNNormalization(tf.keras.layers.Layer):
    """
    article：Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks.
    """

    def __init__(self,
                 epsilon=1e-6,
                 gamma_initializer='ones',
                 beta_initializer='ones',
                 tau_initializer='ones',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 tau_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 tau_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(FPNNormalization, self).__init__(
            name=name, trainable=trainable, **kwargs)
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.tau_initializer = initializers.get(tau_initializer)

        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.tau_regularizer = regularizers.get(tau_regularizer)

        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.tau_constraint = constraints.get(tau_constraint)

    def build(self, input_shape):
        ndims = len(input_shape)
        if ndims is None:
            raise ValueError('Input shape %s has undefined rank.' % input_shape)
        shape = [1] * (ndims - 1)
        shape.append(input_shape[-1])
        self.gamma = self.add_weight(
            shape=shape,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
            name='gamma',
        )
        self.beta = self.add_weight(
            shape=shape,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
            name='beta',
        )

        self.tau = self.add_weight(
            shape=shape,
            initializer=self.tau_initializer,
            regularizer=self.tau_regularizer,
            constraint=self.tau_constraint,
            name='tau',
        )
        self.built = True

    def call(self, x):
        nu2 = math_ops.reduce_mean(math_ops.square(x), axis=[1, 2], keepdims=True)
        x = x * math_ops.sqrt(nu2 + math_ops.abs(self.epsilon))
        return math_ops.maximum(self.gamma * x + self.tau, self.gamma)

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'tau_initializer': initializers.serialize(self.tau_initializer),

            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'tau_regularizer': regularizers.serialize(self.tau_regularizer),

            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'tau_constraint': constraints.serialize(self.tau_constraint),
        }
        base_config = super(FPNNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class BatchNormalization(tf.keras.layers.Layer):
    def __init__(self):
        pass


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-5, name=None, **kwargs):
        super(LayerNorm, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.axis = axis

    def build(self, input_shape):
        n_state = input_shape[-1]
        self.g = self.add_weight(
            shape=[n_state],
            initializer=tf.constant_initializer(1),
            name='g',
        )
        self.b = self.add_weight(
            shape=[n_state],
            initializer=tf.constant_initializer(0),
            name='b',
        )

    def call(self, inputs, ):
        u = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        s = tf.reduce_mean(tf.square(inputs - u), axis=self.axis, keepdims=True)
        x = (inputs - u) * tf.math.rsqrt(s + self.epsilon)
        x = x * self.g + self.b
        return x


if __name__ == "__main__":
    x = tf.constant([[[0.1, 2.], [3., 0.06]], [[3., 0.04], [0.5, 8.]]])
    y = LayerNorm(axis=-1, epsilon=1e-5)
    ll = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
    bb = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5)
    op = ll(x)
    print(op)
    o = y(x)
    print(o)
    k=bb(x)
    print(k)