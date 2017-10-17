import tensorflow as tf
import math


def gaussian_noise_layer(input_tensor, std=0.1):
    noise = tf.random_normal(shape=tf.shape(input_tensor), mean=0.0, stddev=std, dtype=tf.float32)
    return input_tensor + noise
