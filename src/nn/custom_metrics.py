__all__ = [
    "MeanL1RelativeError",
    "MeanL2RelativeError",
    "MaxL1RelativeError",
    "MaxL1AbsoluteError"
]

import tensorflow as tf


class MeanL1RelativeError(tf.keras.metrics.Metric):
    def __init__(self, name="mean_L1_relative_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype=tf.uint64, initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        val = tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true)+tf.keras.backend.epsilon())
        count = tf.cast(tf.shape(y_true)[0], dtype=tf.uint64)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            val *= sample_weight
        self.sum.assign_add(tf.reduce_sum(val))
        self.count.assign_add(count)
    
    def result(self):
        count = tf.cast(self.count, self.dtype)
        return self.sum / count
    
    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)


class MeanL2RelativeError(tf.keras.metrics.Metric):
    def __init__(self, name="mean_L2_relative_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype=tf.uint64, initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        val = tf.math.squared_difference(y_true,y_pred) / (tf.math.square(y_true)+tf.keras.backend.epsilon())
        count = tf.cast(tf.shape(y_true)[0], dtype=tf.uint64)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            val *= sample_weight
        self.sum.assign_add(tf.reduce_sum(val))
        self.count.assign_add(count)
    
    def result(self):
        count = tf.cast(self.count, self.dtype)
        return self.sum / count
    
    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)


class MaxL1RelativeError(tf.keras.metrics.Metric):
    def __init__(self, name="max_L1_relative_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        val = tf.math.abs(y_true-y_pred) / (tf.math.abs(y_true)+tf.keras.backend.epsilon())
        val = tf.reduce_max(val)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            val *= sample_weight
        if val > self.sum:
            self.sum.assign(val)
    
    def result(self):
        return self.sum
    
    def reset_states(self):
        self.sum.assign(0)


class MaxL1AbsoluteError(tf.keras.metrics.Metric):
    def __init__(self, name="max_L1_absolute_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        val = tf.math.abs(y_true-y_pred)
        val = tf.reduce_max(val)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            val *= sample_weight
        if val > self.sum:
            self.sum.assign(val)
    
    def result(self):
        return self.sum
    
    def reset_states(self):
        self.sum.assign(0)

