import tensorflow as tf
import tensorflow.contrib.layers as ly

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from aae import Config, Operation
'''

build layers

'''
config = Config()
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


def decoder_z_x(z, is_test=False):
    if is_test:
        train = z
    else:
        train = opt.gaussian_noise_layer(z)
    train = ly.fully_connected(train, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                               normalizer_params={'fused': True},
                               weights_initializer=tf.random_normal_initializer(0, 0.01))
    train = ly.fully_connected(train, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
                               normalizer_params={'fused': True},
                               weights_initializer=tf.random_normal_initializer(0, 0.01))
    train = ly.fully_connected(train, config.ndim_x, activation_fn=tf.nn.tanh,
                               weights_initializer=tf.random_normal_initializer(0, 0.01))
    return train


def discriminator_z(y, z, reuse=False, is_test=False, apply_softmax=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        img = tf.concat([z, y], axis=-1)
        if not is_test:
            img = opt.gaussian_noise_layer(img)
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


class Object:
    pass


def build_graph(is_test=False):
    # Inputs
    images = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_x])
    conditional_labels = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_y + 1])
    z_sampler = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_z])
    learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

    # Graph
    encoder = encoder_x_z
    decoder = decoder_z_x
    discriminator = discriminator_z

    with tf.variable_scope('encoder'):
        z_representation = encoder(images)
    with tf.variable_scope('decoder'):
        reconstruction = decoder(z_representation, is_test=is_test)

    if is_test:
        return images, z_representation, reconstruction

    probability_fake_sample = discriminator(conditional_labels, z_representation, is_test=is_test)
    probability_true_sample = discriminator(conditional_labels, z_sampler, reuse=True, is_test=is_test)

    # Loss function
    # classification
    # 0 -> true sample
    # 1 -> generated sample
    class_true = tf.ones(shape=(config.batch_size, config.ndim_z / 2), dtype=tf.int32)
    class_fake = tf.zeros(shape=(config.batch_size, config.ndim_z / 2), dtype=tf.int32)
    loss_discriminator = opt.softmax_cross_entropy(probability_fake_sample, probability_true_sample, class_fake,
                                                   class_true)
    loss_encoder = opt.softmax_cross_entropy(probability_fake_sample, probability_true_sample,\
                                             class_fake, class_true, for_generator=True)
    loss_resconstruction = opt.euclidean_distance(images, reconstruction)

    # Variables Collection
    variables_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    variables_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    variables_discriminator = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # Optimizer
    counter_encoder = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_resconstruction = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_discriminator = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)

    opt_resconstruction = opt.optimize(loss_resconstruction, variables_decoder + variables_encoder,
                                       optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                       learning_rate=learning_rate, global_step=counter_resconstruction
                                       )

    opt_discriminator = opt.optimize(loss_discriminator, variables_discriminator,
                                     optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                     learning_rate=learning_rate * 10, global_step=counter_discriminator
                                     )

    opt_encoder = opt.optimize(loss_encoder, variables_encoder,
                               optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                               learning_rate=learning_rate * 10, global_step=counter_encoder
                               )

    # weight Clipping
    clipped_var_discriminator = [
        tf.assign(var, tf.clip_by_value(var, config.gradient_clipping_lower, config.gradient_clipping_upper))
        for var in variables_discriminator]
    with tf.control_dependencies([opt_discriminator]):
        opt_discriminator = tf.tuple(clipped_var_discriminator)

    # output what we want
    graph_handle = Object()
    graph_handle.x = images
    graph_handle.label = conditional_labels
    graph_handle.z = z_sampler

    graph_handle.x_ = reconstruction
    graph_handle.z_r = z_representation

    graph_handle.opt_r = opt_resconstruction
    graph_handle.opt_d = opt_discriminator
    graph_handle.opt_e = opt_encoder
    graph_handle.loss_d = loss_discriminator
    graph_handle.loss_e = loss_encoder
    graph_handle.loss_r = loss_resconstruction
    graph_handle.lr = learning_rate

    return graph_handle
