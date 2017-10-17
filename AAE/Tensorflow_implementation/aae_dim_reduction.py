import tensorflow as tf
import aae
import sequential.utility as util
import sequential.LossFunctions as Loss
import numpy as np


# optimizer for specified variable
class Config(aae.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.ndim_x = 28 ** 2
        self.ndim_y = 10
        self.ndim_reduction = 2
        self.ndim_z = self.ndim_reduction
        self.momentum = 0.5
        self.cluster_head_distance_threshold = 4.5
        self.distribution_z = 'deterministic'
        self.batch_size = 100


def compute_accuracy(self, y_predict, y_true):
    prediction = np.argmax(y_predict, 1)
    equality = np.equal(prediction, y_true)
    return np.mean(equality.astype(np.float32))


def half_euclidean_distance(self, g_start, g_end, is_clipping=False):
    config = Config()
    euclidean_distance = tf.reduce_sum(tf.square(g_start - g_end), axis=1)
    if is_clipping:
        euclidean_distance = tf.minimum(euclidean_distance, tf.fill(
            dims=tf.shape(euclidean_distance), value=tf.square(config.cluster_head_distance_threshold)
        ))
    return tf.reduce_sum(euclidean_distance) / 2


class Operation(aae.Operation):
    half_euclidean_distance = half_euclidean_distance
    nCr = util.nCr
    compute_accuracy = compute_accuracy
