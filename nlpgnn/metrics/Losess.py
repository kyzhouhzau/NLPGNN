import tensorflow as tf


class MaskSparseCategoricalCrossentropy():
    def __init__(self, from_logits=False, use_mask=False):
        self.from_logits = from_logits
        self.use_mask = use_mask

    def __call__(self, y_true, y_predict, input_mask=None):
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_predict, self.from_logits)
        if self.use_mask:
            input_mask = tf.cast(input_mask, dtype=tf.float32)
            input_mask /= tf.reduce_mean(input_mask)
            cross_entropy *= input_mask
            # mask loss
            return tf.reduce_mean(cross_entropy)
        else:
            return tf.reduce_mean(cross_entropy)


class MaskCategoricalCrossentropy():
    def __init__(self, from_logits=False, use_mask=True):
        self.from_logits = from_logits
        self.use_mask = use_mask

    def __call__(self, y_true, y_predict, input_mask=None):
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_predict, self.from_logits)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_true, y_predict)

        if self.use_mask:
            input_mask = tf.cast(input_mask, dtype=tf.float32)
            input_mask /= tf.reduce_mean(input_mask)
            cross_entropy *= input_mask
            return tf.reduce_mean(cross_entropy)
        else:
            return tf.reduce_mean(cross_entropy)

class CategoricalCrossentropy():
    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def __call__(self, y_true, y_predict):
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_predict, self.from_logits)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_true, y_predict)


        return tf.reduce_mean(cross_entropy)
