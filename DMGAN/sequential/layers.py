"""
Created on Jan 11 2018
Author: Nianzu Ethan Zheng
Copyright
"""

import tensorflow as tf
import numpy as np


def concat_label(x, label):
    x_shape = x.get_shape().as_list()
    label_shape = label.get_shape().as_list()
    dx, dl = len(x_shape), len(label_shape)
    if dx == 2 and dl == 2:
        return tf.concat([x, label], axis=1)
    elif dx == 4 and dl == 2:
        label = tf.reshape(label, [-1, 1, 1, label_shape[-1]])
        return tf.concat([x, label * tf.ones_like(x)], axis=-1)
    elif dx == 4 and dl == 4:
        return tf.concat([x, label], axis=-1)
    else:
        raise Exception('Please check your input shape')


def gaussian_noise_layer(input_tensor, std=0.1):
    noise = tf.random_normal(shape=tf.shape(input_tensor), mean=0.0, stddev=std, dtype=tf.float32)
    return input_tensor + noise


def _activation(input, activation=None):
    assert activation in ['relu', 'leaky', 'tanh', 'sigmoid', 'softmax', None]
    if activation == 'relu':
        return tf.nn.relu(input)
    elif activation == 'leaky':
        return tf.contrib.keras.layers.LeakyReLU(0.2)(input)
    elif activation == 'tanh':
        return tf.tanh(input)
    elif activation == 'sigmoid':
        return tf.sigmoid(input)
    elif activation == 'softmax':
        return tf.nn.softmax(input)
    else:
        return input


def mlp(_input, out_features, activation_fn, is_training=True, bias=True, norm=False, name='mlp'):
    with tf.variable_scope(name):
        _, n = _input.get_shape().as_list()
        w = weight_variable_xavier([n, out_features], name='weight')
        out = tf.matmul(_input, w)
        if bias:
            b = bias_variable([out_features], name='bias')
            out = out + b
        out = _activation(out, activation=activation_fn)
        if norm:
            out = batch_norm(out, is_training=is_training)
    return out


def conv2d(_input, out_features, kernel_size,
           strides=None, padding='SAME', bias=False):
    if strides is None:
        strides = [1, 1, 1, 1]
    in_features = int(_input.get_shape()[-1])
    filters = weight_variable_msra(
        [1, kernel_size, in_features, out_features],
        name='kernel')
    out = tf.nn.conv2d(_input, filters, strides=strides, padding=padding)
    if bias:
        b = bias_variable([1, 1, 1, out_features])
        out = out + b
    return out


def deconv2d(_input, kernel_size=3, out_shape=None,
             strides=None, padding='SAME'):
    """out_shape should be spacial size"""
    if strides is None:
        strides = [1, 1, 2, 1]
    n, h, w, in_features = _input.get_shape().as_list()
    num_filters = in_features // 2
    filters = weight_variable_msra(
        [1, kernel_size, num_filters, in_features],
        name='kernel')
    if out_shape is None:
        out_shape = tf.stack([n, 1, w * 2, num_filters])
    else:
        out_shape = tf.stack([n, 1, out_shape[1], num_filters])
    return tf.nn.conv2d_transpose(_input, filters, out_shape, strides, padding=padding)


def flatten(_input):
    return tf.reshape(_input, [-1, np.prod(_input.get_shape().as_list()[1:])])


def max_pool(_input, k):
    ksize = [1, 1, k, 1]
    strides = [1, 1, k, 1]
    padding = 'SAME'
    output = tf.nn.max_pool(_input, ksize, strides, padding)
    return output


def batch_norm(_input, is_training):
    output = tf.contrib.layers.batch_norm(
        _input, scale=True, is_training=is_training)
    return output


def dropout(_input, keep_prob, is_training):
    if keep_prob < 1:
        output = tf.cond(
            is_training,
            lambda: tf.nn.dropout(_input, keep_prob=keep_prob),
            lambda: _input)
    else:
        output = _input
    return output


def weight_variable_msra(shape, name):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.contrib.layers.variance_scaling_initializer()
    )


def weight_variable_xavier(shape, name):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.contrib.layers.variance_scaling_initializer()
    )


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name=name, initializer=initial)

if __name__ == '__main__':
    from sequential.utils import logger
    def log(out):
        logger.info(out.get_shape())

    _input=tf.placeholder(
        dtype=tf.float32,
        shape=[64, 1, 15, 1]
    )
    z = tf.placeholder(
        dtype=tf.float32
        ,shape=[64, 2]
    )
    # z = concat_label(_input, _input)
    # z = concat_label(z, z)
    x = _input
    label = z
    x_shape = x.get_shape().as_list()
    label_shape = label.get_shape().as_list()
    dnum_x = len(x_shape)
    dnum_l = len(label_shape)
    logger.info('{},{}'.format(dnum_x,dnum_l))
    z = concat_label(_input, z)
    log(z)
