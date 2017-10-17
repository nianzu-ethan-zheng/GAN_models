import tensorflow as tf
import tensorflow.contrib.layers as ly
import sys
import os

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


def transform(y, reuse=False):
    with tf.variable_scope('transform') as scope:
        if reuse:
            scope.reuse_variables()
        img = tf.layers.dense(y, config.ndim_reduction, activation=None, use_bias=False,
                              kernel_initializer=tf.random_normal_initializer(0, 1))
        return img


def encoder_yz_representation(y, z, reuse=False):
    return transform(y, reuse=reuse) + z
