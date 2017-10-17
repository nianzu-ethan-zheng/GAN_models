import tensorflow as tf
import tensorflow.contrib.layers as ly

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from aae import Config, Operation
'''

build layers

'''
config = Config()
config.num_types_of_label = 10
opt = Operation()


def encoder_x_z(x):
    img = ly.fully_connected(x, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                             normalizer_params={'fused': True},
                             weights_initializer=tf.random_normal_initializer(0, 0.01))
    img = ly.fully_connected(img, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                             normalizer_params={'fused': True},
                             weights_initializer=tf.random_normal_initializer(0, 0.01))
    if config.distribution_z == 'deterministic':
        img = ly.fully_connected(img, config.ndim_z, activation_fn=None,
                                 weights_initializer=tf.random_normal_initializer(0, 0.01))
    elif config.distribution_z == 'gaussian':
        mu = ly.fully_connected(img, config.ndim_z, activation_fn=None,
                                weights_initializer=tf.random_normal_initializer(0, 0.01))
        log_sigma = ly.fully_connected(img, config.ndim_z, activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer(0, 0.01))
        eps = tf.random_normal(shape=tf.shape(mu), mean=0, stddev=1)
        img = mu + eps * log_sigma
    else:
        raise Exception()
    return img


def decoder_z_x(z):
    train = ly.fully_connected(z, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                               normalizer_params={'fused': True},
                               weights_initializer=tf.random_normal_initializer(0, 0.01))
    train = ly.fully_connected(train, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                               normalizer_params={'fused': True},
                               weights_initializer=tf.random_normal_initializer(0, 0.01))
    train = ly.fully_connected(train, config.ndim_x, activation_fn=tf.nn.tanh,
                               weights_initializer=tf.random_normal_initializer(0, 0.01))
    return train


def discriminator_z(z, reuse=False, apply_softmax=False):
    with tf.variable_scope('discriminator_z') as scope:
        if reuse:
            scope.reuse_variables()

        img = z
        img = ly.fully_connected(img, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                                 normalizer_params={'fused': True},
                                 weights_initializer=tf.random_normal_initializer(0, 0.01))
        img = ly.fully_connected(img, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                                 normalizer_params={'fused': True},
                                 weights_initializer=tf.random_normal_initializer(0, 0.01))
        if apply_softmax:
            img = ly.fully_connected(img, 2, activation_fn=tf.nn.softmax,
                                     weights_initializer=tf.random_normal_initializer(0, 0.01))
        else:
            img = ly.fully_connected(img, 2, activation_fn=None,
                                     weights_initializer=tf.random_normal_initializer(0, 0.01))
        return img


def discriminator_img(img, reuse=False, apply_softmax=False):
    with tf.variable_scope('discriminator_img') as scope:
        if reuse:
            scope.reuse_variables()

        img = ly.fully_connected(img, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                                 normalizer_params={'fused': True},
                                 weights_initializer=tf.random_normal_initializer(0, 0.01))
        img = ly.fully_connected(img, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                                 normalizer_params={'fused': True},
                                 weights_initializer=tf.random_normal_initializer(0, 0.01))
        if apply_softmax:
            img = ly.fully_connected(img, 2, activation_fn=tf.nn.softmax,
                                     weights_initializer=tf.random_normal_initializer(0, 0.01))
        else:
            img = ly.fully_connected(img, 2, activation_fn=None,
                                     weights_initializer=tf.random_normal_initializer(0, 0.01))
        return img
