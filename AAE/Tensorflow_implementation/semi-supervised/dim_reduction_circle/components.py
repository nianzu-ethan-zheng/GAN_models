import tensorflow as tf
import tensorflow.contrib.layers as ly
import sys
import os
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from aae_dim_reduction import Config, Operation

config = Config()
opt = Operation()


def decoder_zy_x(z):
    train = ly.fully_connected(z, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                               normalizer_params={'fused': True},
                               weights_initializer=tf.random_normal_initializer(0, config.weight_std))
    train = ly.fully_connected(train, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                               normalizer_params={'fused': True},
                               weights_initializer=tf.random_normal_initializer(0, config.weight_std))
    train = ly.fully_connected(train, config.ndim_x, activation_fn=tf.nn.tanh,
                               weights_initializer=tf.random_normal_initializer(0, config.weight_std))
    return train


def encoder_x_yz(x):
    img = ly.fully_connected(x, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                             normalizer_params={'fused': True},
                             weights_initializer=tf.random_normal_initializer(0, config.weight_std))
    img = ly.fully_connected(img, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                             normalizer_params={'fused': True},
                             weights_initializer=tf.random_normal_initializer(0, config.weight_std))
    # z part
    if config.distribution_z == 'deterministic':
        z = ly.fully_connected(img, config.ndim_z, activation_fn=None,
                               weights_initializer=tf.random_normal_initializer(0, config.weight_std))
    elif config.distribution_z == 'gaussian':
        mu = ly.fully_connected(img, config.ndim_z, activation_fn=None,
                                weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        log_sigma = ly.fully_connected(img, config.ndim_z, activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        eps = tf.random_normal(shape=tf.shape(mu), mean=0, stddev=1)
        z = mu + eps * log_sigma
    else:
        raise Exception()

    # y part

    y_logits = ly.fully_connected(img, config.ndim_y, activation_fn=None,
                                  weights_initializer=tf.random_normal_initializer(0, config.weight_std))
    y = tf.nn.softmax(y_logits)
    return y, z, y_logits


def discriminator_z(z, reuse=False, is_test=False, apply_softmax=False):
    with tf.variable_scope('discriminator_z') as scope:
        if reuse:
            scope.reuse_variables()

        if not is_test:
            z = opt.gaussian_noise_layer(z)
        img = ly.fully_connected(z, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                                 normalizer_params={'fused': True},
                                 weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        img = ly.fully_connected(img, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                                 normalizer_params={'fused': True},
                                 weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        if apply_softmax:
            img = ly.fully_connected(img, 2, activation_fn=tf.nn.softmax,
                                     weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        else:
            img = ly.fully_connected(img, 2, activation_fn=None,
                                     weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        return img


def discriminator_y(y, reuse=False, is_test=False, apply_softmax=False):
    with tf.variable_scope('discriminator_y') as scope:
        if reuse:
            scope.reuse_variables()

        if not is_test:
            y = opt.gaussian_noise_layer(y)
        img = ly.fully_connected(y, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                                 normalizer_params={'fused': True},
                                 weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        img = ly.fully_connected(img, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                                 normalizer_params={'fused': True},
                                 weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        if apply_softmax:
            img = ly.fully_connected(img, 2, activation_fn=tf.nn.softmax,
                                     weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        else:
            img = ly.fully_connected(img, 2, activation_fn=None,
                                     weights_initializer=tf.random_normal_initializer(0, config.weight_std))
        return img


def transform(y):
    if not hasattr(config, 'wc'):
        r = 8
        wc = np.zeros([config.ndim_y, config.ndim_reduction],dtype=np.float32)
        for m in range(config.ndim_y):
            wc[m] = [r * math.cos(np.pi * 2 / config.ndim_y * m), r * math.sin(np.pi * 2 / config.ndim_y * m)]
        config.wc = wc
    return tf.matmul(y, config.wc)


def encoder_yz_representation(y, z):
    return transform(y) + z
