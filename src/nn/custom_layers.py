__all__ = ["Standarization", "Normalization"]

import os
import numpy as np
import tensorflow as tf


class Normalization(tf.keras.layers.Layer):
    def load_bounds(self, pmin, pmax):
        self.pmin = pmin
        self.pmax = pmax
        self.dp = self.pmax - self.pmin

    def call(self, p):
        return (p - self.pmin) / self.dp


class Standarization(tf.keras.layers.Layer):
    def adapt(self, p_sample):
        self.means_ = tf.convert_to_tensor(np.mean(p_sample, axis=0, keepdims=True))
        self.stds_ = tf.convert_to_tensor(np.std(p_sample, axis=0, keepdims=True))
    
    def load_params(self, means, stds):
        self.means_ = tf.convert_to_tensor(means)
        self.stds_ = tf.convert_to_tensor(stds)
    
    def save_params(self, path):
        np.save(os.path.join(path, "mean.npy"), self.means_.numpy())
        np.save(os.path.join(path, "std.npy"), self.stds_.numpy())
    
    def call(self, p):
        return (p - self.means_) / (self.stds_+tf.keras.backend.epsilon())