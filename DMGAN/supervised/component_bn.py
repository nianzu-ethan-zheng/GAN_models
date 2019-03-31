"""
Author : Nianzu Ethan Zheng
Datetime: 2018-1-25
Place: Shenyang, china
Copyright.
"""
import os, sys
import tensorflow as tf
from testbed import process

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from base_options import BaseConfig
from sequential.layers import concat_label
from ru_net import ru_net
from encoder import encoder
from sequential.block_provider import composite_function
from sequential.layers import max_pool, mlp, flatten
from sequential.utils import logger, log

config = BaseConfig()


@log
def encoder_x_z(_input, is_training=config.is_train, reuse=False):
    with tf.variable_scope('encoder') as scope:
        if reuse:
            scope.reuse_variables()
        out = encoder(_input, is_training, latent=config.distribution_z)
    return out


@log
def decoder_z_x(_input, z, c, is_training=config.is_train, reuse=False):
    with tf.variable_scope('decoder') as scope:
        if reuse:
            scope.reuse_variables()
        z = concat_label(z, c)
        _input = concat_label(_input, z)
        net = ru_net()
        out = net.build_ru_net(_input, is_training)
    return out


@log
def discriminator_z(z, reuse=False, is_training=config.is_train, apply_softmax=False):
    with tf.variable_scope('discriminator_z') as scope:
        if reuse:
            scope.reuse_variables()
        out = mlp(z, 64, 'leaky', is_training, norm=True, name='mlp_1')
        out = mlp(out, 64, 'leaky', is_training, norm=True, name='mlp_2')
        if apply_softmax:
            out = mlp(out, 2, 'softmax', is_training, name='mlp_3')
        else:
            out = mlp(out, 2, None, is_training, name='mlp_3')
        return out


@log
def discriminator_img(_input, z, c, reuse=False, is_training=config.is_train, apply_softmax=False):
    with tf.variable_scope('discriminator_img') as scope:
        if reuse:
            scope.reuse_variables()

        num_filters = [64, 128, 256]
        z = concat_label(z, c)
        out = concat_label(_input, z)
        for num, filters in enumerate(num_filters):
            with tf.variable_scope('layer_%d' % num):
                out = composite_function(out, filters, is_training=is_training)
                out = max_pool(out, k=2)
        out = flatten(out)
        out = mlp(out, 512, 'relu', is_training, norm=True, name='mlp_1')
        if apply_softmax:
            out = mlp(out, 2, 'softmax', is_training, name='mlp_2')
        else:
            out = mlp(out, 2, None, is_training, name='mlp_2')
        return out


@log
def process_x(_input, is_training=config.is_train, reuse=False):
    with tf.variable_scope('process') as scope:
        if reuse:
            scope.reuse_variables()
        out = process(_input, is_training, latent='dm')
    return out


if __name__ == "__main__":
    img = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, 1, config.ndim_x, 1]
    )
    z_prior = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, config.ndim_z]
    )
    z_ept = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, config.ndim_z]
    )
    z_lat = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, config.ndim_z]
    )
    img_prior = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, 1, config.ndim_x, 1]
    )
    img_cond = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, 1, config.ndim_x, 1]
    )
    pt = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, config.ndim_z]
    )
    pi_int = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, 1, config.ndim_x, 1]
    )
    print("A model is being built")
    # z_img = encoder_x_z(img)
    # def log(out):
    #     logger.info(out.get_shape())
    # log(concat_label(z_ept, z_prior))
    # log(concat_label(img, z_ept))
    #
    #
    # img_z = decoder_z_x(img_cond, z_ept, z_ept)
    #
    # [logger.info(var) for var in tf.trainable_variables()]
    #
    # img_latent = decoder_z_x(img_cond, z_lat, z_ept, reuse=True)
    # z_let = encoder_x_z(img_latent, reuse=True)
    # process
    # with tf.variable_scope('get_the_optimal_adjustment'):
    #     r, d = tf.split(
    #         img, [config.ndim_r, config.ndim_d], axis=2)
    #     rd = tf.concat(
    #         [r, tf.zeros_like(d)], axis=2)
    #     pi = pi_int + rd
    # logger.info(r.get_shape)
    # with tf.variable_scope('process'):
    po = process_x(img)

    [logger.info(var) for var in tf.trainable_variables()]
    # dq = po - z_lat
    # logger.info(z_prior.get_shape())
    # D_z = discriminator_z(
    #     z_prior
    # )
    # [logger.info(var) for var in tf.trainable_variables()]
    # tf.trainable_variables()
    # logger.info(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_z'))
    # D_z_prior = discriminator_z(
    #     z_prior,
    #     reuse=True
    # )
    # D_img = discriminator_img(
    #     img_cond, z_lat, z_ept
    # )
    # [logger.info(var) for var in tf.trainable_variables()]
    # D_img_prior = discriminator_img(
    #     img_cond, img_prior, z_ept, reuse=True
    # )
