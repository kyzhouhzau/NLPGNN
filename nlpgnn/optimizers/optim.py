#! usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf


class AdamWarmup(tf.keras.optimizers.Optimizer):
    def __init__(self,
                 learning_rate=0.01,  # 重要参数
                 decay_steps=10e10,  # 重要参数
                 warmup_steps=0,  # 重要参数 0.1*decay_steps
                 end_learning_rate=0.0,
                 power=1.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 weight_decay_rate=0.01,
                 epsilon=1e-8,
                 bias_correction=True,
                 weight_decay_pattern=["LayerNorm", "layer_norm", "bias"],
                 name='Adam',
                 **kwargs):
        """
        :param warmup_steps: 学习率在指定的步数线性增长到目标学习率
        :param learning_rate:
        :param beta_1:
        :param beta_2:
        :param epsilon:
        :param bias_correction:
        :param name:
        :param kwargs:
        参考：
        https://github.com/CyberZHG/keras-bert
        https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/backend.py#L1561-L1563
        https://github.com/bojone/bert4keras
        """
        super(AdamWarmup, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('weight_decay_rate', weight_decay_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('power', power)
        self._set_hyper('warmup_steps', warmup_steps)
        self._set_hyper('decay_steps', decay_steps)
        self._set_hyper('end_learning_rate', end_learning_rate)

        self.epsilon = epsilon or tf.keras.backend.epislon()
        self.bias_correction = bias_correction  # 是否做偏差修正
        self.weight_decay_rate = weight_decay_rate
        self.weight_decay_pattern = weight_decay_pattern

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _resource_apply_op(self, grad, var, indices=None):
        # 准备变量
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        warmup_steps = self._get_hyper('warmup_steps', var_dtype)
        epsilon_t = tf.cast(self.epsilon, var_dtype)
        global_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t_power = tf.math.pow(beta_1_t, global_step)
        beta_2_t_power = tf.math.pow(beta_2_t, global_step)

        decay_steps = self._get_hyper('decay_steps', var_dtype)
        end_learning_rate = self._get_hyper('end_learning_rate', var_dtype)
        power = self._get_hyper('power', var_dtype)

        # Warmup + 多项式损失
        # if global_step <= warmup_steps and warmup_steps > 0:
        #     lr_t = self.learning_rate * (global_step / warmup_steps)
        # else:
        #     if decay_steps > 0:
        #         lr_t = end_learning_rate + (lr_t - end_learning_rate) * \
        #                (1.0 - tf.minimum(global_step, decay_steps) / decay_steps) ** (power)

        # Warmup + 多项式损失 (use tf.function)
        lr_t = tf.where(
            global_step <= warmup_steps,
            lr_t * (global_step / warmup_steps),
            end_learning_rate + (lr_t - end_learning_rate) * (
                    1.0 - tf.minimum(global_step, decay_steps) / decay_steps) ** (power),
        )

        if indices is None:
            # update 与 state_ops.assign(x,new_x)相同均是赋值含义
            m_t = tf.keras.backend.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = tf.keras.backend.update(v, beta_2_t * v + (1 - beta_2_t) * grad ** 2)
        else:
            mv_ops = [tf.keras.backend.update(m, beta_1_t * m), tf.keras.backend.update(v, beta_2_t * v)]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(m, indices,
                                                 (1 - beta_1_t) * grad)
                v_t = self._resource_scatter_add(v, indices,
                                                 (1 - beta_2_t) * grad ** 2)

        with tf.control_dependencies([m_t, v_t]):  # 控制计算流图，当[m_t, v_t执行完成后，才执行以下内容]
            # 偏差修正
            if self.bias_correction:
                m_t = m_t / (1. - beta_1_t_power)
                v_t = v_t / (1. - beta_2_t_power)
            var_t = m_t / (tf.keras.backend.sqrt(v_t) + epsilon_t)

            # weight decay
            # if self.weight_decay_rate > 0.0:
            weight_decay_rate = self._get_hyper('weight_decay_rate', var_dtype)
            if self.weight_decay_pattern is None:
                var_t += weight_decay_rate * var
            else:
                for pattern in self.weight_decay_pattern:
                    if pattern in var.name:
                        var_t += weight_decay_rate * var
                        break
            var_t = var - lr_t * var_t
            return tf.keras.backend.update(var, var_t)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply_op(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply_op(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'warmup_steps': self._serialize_hyperparameter('warmup_steps'),
            'weight_decay_rate': self._serialize_hyperparameter('weight_decay_rate'),
            'decay_steps': self._serialize_hyperparameter('decay_steps'),
            'end_learning_rate': self._serialize_hyperparameter('end_learning_rate'),
            'power': self._serialize_hyperparameter('power'),
        }
        base_config = super(AdamWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RAdam(tf.keras.optimizers.Optimizer):
    def __init__(self,
                 learning_rate=0.001,  # 重要参数
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 freedom=5,
                 weight_decay_rate=0.,
                 weight_decay_pattern=["LayerNorm", "layer_norm", "bias"],
                 name='RAdam',
                 **kwargs):
        """
        :param warmup_steps: 学习率在指定的步数线性增长到目标学习率
        :param learning_rate:
        :param beta_1:
        :param beta_2:
        :param epsilon:
        :param bias_correction:
        :param name:
        :param kwargs:
        参考：
        https://github.com/CyberZHG/keras-bert
        https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/backend.py#L1561-L1563
        Paper: https://arxiv.org/pdf/1908.03265v1.pdf
        """
        super(RAdam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('freedom', freedom)
        self._set_hyper('weight_decay_rate', weight_decay_rate)
        self.epsilon = epsilon or tf.keras.backend.epislon()
        self.weight_decay_pattern = weight_decay_pattern
        self.weight_decay_rate = weight_decay_rate

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _resource_apply_op(self, grad, var, indices=None):
        # 准备变量
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        freedom = self._get_hyper('freedom', var_dtype)
        epsilon_t = tf.cast(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t_power = tf.math.pow(beta_1_t, local_step)
        beta_2_t_power = tf.math.pow(beta_2_t, local_step)
        sma_inf = 2 / (1 - beta_2_t) - 1
        if indices is None:
            # update 与 state_ops.assign(x,new_x)相同均是赋值含义
            m_t = tf.keras.backend.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = tf.keras.backend.update(v, beta_2_t * v + (1 - beta_2_t) * grad ** 2)
        else:
            mv_ops = [tf.keras.backend.update(m, beta_1_t * m), tf.keras.backend.update(v, beta_2_t * v)]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(m, indices,
                                                 (1 - beta_1_t) * grad)
                v_t = self._resource_scatter_add(v, indices,
                                                 (1 - beta_2_t) * grad ** 2)
        with tf.control_dependencies([m_t, v_t]):
            m_t_corrected = m_t / (1 - beta_1_t_power)
            sma_t = sma_inf - 2.0 * local_step / (1 - beta_2_t_power)
            if sma_t >= freedom:
                v_t_corrected = tf.math.sqrt(v_t / (1 - beta_2_t_power))
                r_t = tf.math.sqrt((sma_t - 4) / (sma_inf - 4) * (sma_t - 2) / (sma_inf - 2) * (sma_inf) / (sma_t))
                var_t = r_t * m_t_corrected / (v_t_corrected + epsilon_t)
            else:
                var_t = m_t_corrected

            # weight decay
            if self.weight_decay_rate > 0.0:
                weight_decay_rate = self._get_hyper('weight_decay_rate', var_dtype)
                if self.weight_decay_pattern is None:
                    var_t += weight_decay_rate * var
                else:
                    for pattern in self.weight_decay_pattern:
                        if pattern in var.name:
                            var_t += weight_decay_rate * var
                            break
            var_t = var - lr_t * var_t
            return tf.keras.backend.update(var, var_t)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply_op(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply_op(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'freedom': self._serialize_hyperparameter('freedom'),
            'epsilon': self.epsilon,
        }
        base_config = super(AdamWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

