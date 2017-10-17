from component_without_bn import *


class Object:
    pass


def build_graph(is_test=False):
    # Inputs
    images = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_x])
    z_sampler = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_z])
    learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

    # Graph
    encoder = encoder_x_z
    decoder = decoder_z_x
    discriminator = discriminator_z

    with tf.variable_scope('encoder'):
        z_representation = encoder(images)
    with tf.variable_scope('decoder'):
        reconstruction = decoder(z_representation)

    if is_test:
        test_handle = Object()
        test_handle.x = images
        test_handle.z_r = z_representation
        test_handle.x_r = reconstruction
        return test_handle

    probability_fake_sample = discriminator(z_representation)
    probability_true_sample = discriminator(z_sampler, reuse=True)

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

    opt_discriminator = opt.optimize(config.scale_ratio * loss_discriminator, variables_discriminator,
                                     optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                     learning_rate=learning_rate, global_step=counter_discriminator
                                     )

    opt_encoder = opt.optimize(config.scale_ratio * loss_encoder, variables_encoder,
                               optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                               learning_rate=learning_rate, global_step=counter_encoder
                               )

    # output what we want
    graph_handle = Object()
    graph_handle.x = images
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
