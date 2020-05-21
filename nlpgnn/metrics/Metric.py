# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements F scores."""

import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, accuracy_score
from typeguard import typechecked

from .type import AcceptableDTypes, FloatTensorLike

warnings.filterwarnings('ignore')


class FBetaScore(tf.keras.metrics.Metric):
    """Computes F-Beta score.
    It is the weighted harmonic mean of precision
    and recall. Output range is [0, 1]. Works for
    both multi-class and multi-label classification.
    F-Beta = (1 + beta^2) * (prec * recall) / ((beta^2 * prec) + recall)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-Beta Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
        ValueError: If the `beta` value is less than or equal
        to 0.
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    @typechecked
    def __init__(
            self,
            num_classes: FloatTensorLike,
            average: Optional[str] = None,
            beta: FloatTensorLike = 1.0,
            threshold: Optional[FloatTensorLike] = None,
            name: str = "fbeta_score",
            dtype: AcceptableDTypes = None,
            **kwargs
    ):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Unknown average type. Acceptable values "
                "are: [None, micro, macro, weighted]"
            )

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float")

        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError("The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    # TODO: Add sample_weight support, currently it is
    # ignored during calculations.
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, shape=[-1])
        y_true = tf.one_hot(y_true, self.num_classes)
        y_pred = tf.reshape(y_pred, shape=[-1, self.num_classes])

        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        def _count_non_zero(val):
            non_zeros = tf.math.count_nonzero(val, axis=self.axis)
            return tf.cast(non_zeros, self.dtype)

        self.true_positives.assign_add(_count_non_zero(y_pred * y_true))
        self.false_positives.assign_add(_count_non_zero(y_pred * (y_true - 1)))
        self.false_negatives.assign_add(_count_non_zero((y_pred - 1) * y_true))
        self.weights_intermediate.assign_add(_count_non_zero(y_true))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)
            )
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
        }

        if self.threshold is not None:
            config["threshold"] = self.threshold

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_states(self):
        self.true_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_negatives.assign(tf.zeros(self.init_shape, self.dtype))
        self.weights_intermediate.assign(tf.zeros(self.init_shape, self.dtype))


class KSparseF1Score(FBetaScore):
    """Computes F-1 Score.
    It is the harmonic mean of precision and recall.
    Output range is [0, 1]. Works for both multi-class
    and multi-label classification.
    F-1 = 2 * (precision * recall) / (precision + recall)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro`
            and `weighted`. Default value is None.
        threshold: Elements of `y_pred` above threshold are
            considered to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-1 Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    @typechecked
    def __init__(
            self,
            num_classes: FloatTensorLike,
            average: str = None,
            threshold: Optional[FloatTensorLike] = None,
            name: str = "f1_score",
            dtype: AcceptableDTypes = None,
            **kwargs
    ):
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


class SparseF1Score(object):
    def __init__(self, average, predict_sparse=False):
        self.average = average
        self.predict_sparse = predict_sparse

    def __call__(self, y_true, y_predict):
        y_true = tf.reshape(tf.constant(y_true), [-1]).numpy()

        if self.predict_sparse:
            y_predict = tf.reshape(y_predict, [-1]).numpy()
        else:
            y_predict = tf.reshape(tf.argmax(y_predict, -1), [-1]).numpy()

        f1 = f1_score(y_true, y_predict, average=self.average)
        return f1


class SparseRecallScore(object):
    def __init__(self, average, predict_sparse=False):
        self.average = average
        self.predict_sparse = predict_sparse

    def __call__(self, y_true, y_predict):
        if self.predict_sparse:
            y_predict = tf.reshape(y_predict, [-1]).numpy()
        else:
            y_predict = tf.reshape(tf.argmax(y_predict, -1), [-1]).numpy()
        y_true = tf.reshape(tf.constant(y_true), [-1]).numpy()
        recall = recall_score(y_true, y_predict, average=self.average)
        return recall


class SparsePrecisionScore(object):
    def __init__(self, average, predict_sparse=False):
        self.average = average
        self.predict_sparse = predict_sparse

    def __call__(self, y_true, y_predict):
        y_true = tf.reshape(tf.constant(y_true), [-1]).numpy()
        if self.predict_sparse:
            y_predict = tf.reshape(y_predict, [-1]).numpy()

        else:
            y_predict = tf.reshape(tf.argmax(y_predict, -1), [-1]).numpy()
        precision = precision_score(y_true, y_predict, average=self.average)
        return precision


class SparseClassficationReport(object):
    def __init__(self, predict_sparse=False):
        self.predict_sparse = predict_sparse

    def __call__(self, y_true, y_predict, labels=None):
        y_true = tf.reshape(tf.constant(y_true), [-1]).numpy()
        if self.predict_sparse:
            y_predict = tf.reshape(y_predict, [-1]).numpy()

        else:
            y_predict = tf.reshape(tf.argmax(y_predict, -1), [-1]).numpy()
        precision = classification_report(y_true, y_predict, labels=labels)
        return precision


class SparseAccuracy(object):
    def __init__(self, predict_sparse=False):
        self.predict_sparse = predict_sparse

    def __call__(self, y_true, y_predict):
        y_true = tf.reshape(tf.constant(y_true), [-1]).numpy()
        if self.predict_sparse:
            y_predict = tf.reshape(y_predict, [-1]).numpy()

        else:
            y_predict = tf.reshape(tf.argmax(y_predict, -1), [-1]).numpy()
        acc = accuracy_score(y_true, y_predict)
        return acc


class MaskAccuracy:
    def __call__(self, labels, predicts, input_mask=None):
        correct_prediction = tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float64)
        mask = tf.cast(input_mask, dtype=tf.float64)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)


class HitN_MR_MRR():
    def __init__(self, loader, mode="valid"):
        self.mode = mode
        self.loader = loader
        if self.mode == "train":
            self.data_idxs = loader.get_data_idxs(loader.train_data)
        elif self.mode == "valid":
            self.data_idxs = loader.get_data_idxs(loader.valid_data)  # valid数据集合中的三元组
        elif self.mode == "test":
            self.data_idxs = loader.get_data_idxs(loader.test_data)
        # er_vocb={(h,r):[t1,t2]},train+valid
        self.er_vocab = loader.get_er_vocab(loader.get_data_idxs(loader.data))

    def __call__(self, model, batch_size):
        ranks = []
        hits = defaultdict(list)
        for i in range(10):
            hits[i] = []
        for data in self.loader.get_batch(self.er_vocab, self.data_idxs, batch_size=batch_size):
            # actually here h_r is h_r_t
            h_idx = data['h_r'][:, 0]
            r_idx = data['h_r'][:, 1]
            t_idx = data['h_r'][:, 2]
            hrt = data["h_r"].numpy()
            predict = model.predict(h_idx, r_idx)
            predict = predict.numpy()
            for j in range(hrt.shape[0]):
                # 也就是说对于当前样本仅仅考虑目标尾实体的得分，而其他的也是真实尾实体得分被置为0。
                filt = self.er_vocab[(hrt[j][0], hrt[j][1])]
                target_value = predict[j, t_idx[j]]
                predict[j, filt] = 0.0
                predict[j, t_idx[j]] = target_value
            sort_idxs = tf.argsort(predict, axis=1, direction='DESCENDING')
            del predict
            for j in range(hrt.shape[0]):
                rank = np.where(sort_idxs[j] == t_idx[j])[0][0]
                ranks.append(rank + 1)
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
            del sort_idxs

        hit10 = np.mean(hits[9])
        hit5 = np.mean(hits[4])
        hit3 = np.mean(hits[2])
        hit1 = np.mean(hits[0])
        MR = np.mean(ranks)
        MRR = np.mean(1. / np.array(ranks))
        return hit1, hit3, hit5, hit10, MR, MRR
