"""
Discription: A Python Progress Meter
Author: Nianzu Ethan Zheng
Date: 2018-1-21
Copyright
"""

import tensorflow as tf


def wasserstein_distance(f_fake, f_true, for_generator=False):
    if for_generator:
        return tf.reduce_mean(-f_fake)
    else:
        return tf.reduce_mean(f_fake - f_true)


def euclidean_distance(x, x_construction):
    euclidean_distance = tf.reduce_sum(tf.squared_difference(x, x_construction))
    return tf.reduce_mean(euclidean_distance) / 2


def jensen_shannon_divergence(d_fake, d_true, for_generator=False):
    # d_fake                   ---> linear output
    tiny = 1e-10
    if for_generator:
        return -tf.reduce_mean(tf.log(d_fake + tiny))
    else:
        return -tf.reduce_mean(tf.log(d_true + tiny) + tf.log(1. - d_fake + tiny))


def sigmoid_cross_entropy(d_fake, d_true, for_generator=False):
    # d_fake                   ---> linear output
    if for_generator:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_true)))
    else:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_true, labels=tf.ones_like(d_true)) + \
                              tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))


def softmax_cross_entropy(d_fake, d_true, class_fake, class_true, depth=2, for_generator=False):
    labels_true = tf.one_hot(class_true, depth=depth, on_value=1, off_value=0)
    labels_fake = tf.one_hot(class_fake, depth=depth, on_value=1, off_value=0)
    if for_generator:
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_fake, labels=labels_true))
    else:
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_true, labels=labels_true) + \
                              tf.nn.softmax_cross_entropy_with_logits(logits=d_fake, labels=labels_fake))






