import tensorflow as tf
import tensorflow.contrib.layers as ly
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from aae_classification import Config, Operation

'''

build layers

'''
config = Config()
opt = Operation()
config.ndim_x = 28 ** 2
config.ndim_y = 10
config.ndim_z = 10
config.distribution_z = 'deterministic'
config.weight_std = 0.01
config.weight_initializer = ly.batch_norm
config.nonlinearity = tf.nn.relu
config.batch_size = 100
config.learning_rate = 0.0001
config.momentum = 0.1
config.gradient_clipping_upper = 5
config.gradient_clipping_lower = -5
config.weight_decay = 0
config.device = '/gpu:0'
config.ckpt_dir = './ckpt_dir'
config.log_dir = './log_dir'


def decoder_zy_x(y, z):
    train = tf.concat([y, z], axis=-1)
    train = ly.fully_connected(train, 1000, activation_fn=config.nonlinearity, normalizer_fn=ly.batch_norm,
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


class Object:
    pass


def build_graph(is_test=False):
    # Inputs
    images = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_x])
    y_sampler = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_y])
    z_sampler = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_z])
    learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
    y_supervised = tf.placeholder(dtype=tf.int32, shape=[None, ])

    # Graph
    with tf.variable_scope('encoder'):
        img_y, img_z, y_logits = encoder_x_yz(images)
    with tf.variable_scope('decoder'):
        reconstruction = decoder_zy_x(img_y, img_z)

    if is_test:
        return images, img_y, img_z, reconstruction

    prob_fake_y = discriminator_y(img_y, is_test=is_test)
    prob_true_y = discriminator_y(y_sampler, reuse=True, is_test=is_test)
    prob_fake_z = discriminator_z(img_z, is_test=is_test)
    prob_true_z = discriminator_z(z_sampler, reuse=True, is_test=is_test)

    # Loss function
    # classification
    # 0 -> true sample
    # 1 -> generated sample
    class_true = tf.ones(shape=(config.batch_size, 1), dtype=tf.int32)
    class_fake = tf.zeros(shape=(config.batch_size, 1), dtype=tf.int32)
    loss_discriminator_y = opt.softmax_cross_entropy(prob_fake_y, prob_true_y, class_fake, class_true)
    loss_generator_y = opt.softmax_cross_entropy(prob_fake_y, prob_true_y, class_fake, class_true, for_generator=True)
    loss_discriminator_z = opt.softmax_cross_entropy(prob_fake_z, prob_true_z, class_fake, class_true)
    loss_generator_z = opt.softmax_cross_entropy(prob_fake_z, prob_true_z, class_fake, class_true, for_generator=True)
    loss_resconstruction = opt.euclidean_distance(images, reconstruction)
    loss_encoder = loss_generator_y + loss_generator_z
    # supervised way
    y_onehot_supervised = tf.one_hot(y_supervised, depth=10, on_value=1, off_value=0)
    loss_encoder_supervised = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_onehot_supervised))

    # Variables Collection
    variables_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    variables_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    variables_discriminator_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_y')
    variables_discriminator_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_z')

    # Optimizer
    counter_encoder = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_encoder_supervised = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_resconstruction = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_discriminator_y = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_discriminator_z = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)

    opt_resconstruction = opt.optimize(loss_resconstruction, variables_decoder + variables_encoder,
                                       optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                       learning_rate=learning_rate, global_step=counter_resconstruction
                                       )

    opt_discriminator_y = opt.optimize(loss_discriminator_y, variables_discriminator_y,
                                       optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                       learning_rate=learning_rate, global_step=counter_discriminator_y
                                       )

    opt_discriminator_z = opt.optimize(loss_discriminator_z, variables_discriminator_z,
                                       optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                       learning_rate=learning_rate, global_step=counter_discriminator_z
                                       )

    opt_encoder = opt.optimize(loss_encoder, variables_encoder,
                               optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                               learning_rate=learning_rate, global_step=counter_encoder
                               )

    opt_encoder_supervised = opt.optimize(loss_encoder_supervised, variables_encoder,
                                          optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                          learning_rate=learning_rate, global_step=counter_encoder_supervised
                                          )

    # weight Clipping
    clipped_var_discriminator_y = [
        tf.assign(var, tf.clip_by_value(var, config.gradient_clipping_lower, config.gradient_clipping_upper))
        for var in variables_discriminator_y]
    with tf.control_dependencies([opt_discriminator_y]):
        opt_discriminator_y = tf.tuple(clipped_var_discriminator_y)

    clipped_var_discriminator_z = [
        tf.assign(var, tf.clip_by_value(var, config.gradient_clipping_lower, config.gradient_clipping_upper))
        for var in variables_discriminator_z]
    with tf.control_dependencies([opt_discriminator_z]):
        opt_discriminator_z = tf.tuple(clipped_var_discriminator_z)

    # output what we want
    graph_handle = Object()
    graph_handle.x = images
    graph_handle.y = y_sampler
    graph_handle.y_s = y_supervised
    graph_handle.z = z_sampler
    graph_handle.x_ = reconstruction
    graph_handle.y_r = img_y
    graph_handle.z_r = img_z
    # operation
    graph_handle.opt_r = opt_resconstruction
    graph_handle.opt_dy = opt_discriminator_y
    graph_handle.opt_dz = opt_discriminator_z
    graph_handle.opt_e = opt_encoder
    graph_handle.opt_ey = opt_encoder_supervised
    # losses
    graph_handle.loss_dy = loss_discriminator_y
    graph_handle.loss_dz = loss_discriminator_z
    graph_handle.loss_gy = loss_generator_y
    graph_handle.loss_gz = loss_generator_z
    graph_handle.loss_ey = loss_encoder_supervised
    graph_handle.loss_r = loss_resconstruction
    graph_handle.lr = learning_rate

    return graph_handle
