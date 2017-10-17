from component_without_bn import *


class Object:
    pass


def build_graph(is_test=False):
    # Inputs
    images = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_x])
    z_sampler = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_z])
    learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
    img_sampler = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_x])

    # Graph
    with tf.variable_scope('encoder'):
        z_representation = encoder_x_z(images)
    with tf.variable_scope('decoder'):
        reconstruction = decoder_z_x(z_representation)

    if is_test:
        test_handle = Object()
        test_handle.x = images
        test_handle.z_r = z_representation
        test_handle.x_r = reconstruction
        return test_handle

    D_z = discriminator_z(z_representation)
    D_z_prior = discriminator_z(z_sampler, reuse=True)
    D_img = discriminator_img(reconstruction)
    D_img_prior = discriminator_img(img_sampler, reuse=True)

    # Loss function
    # classification
    # 0 -> true sample
    # 1 -> generated sample
    class_true = tf.ones(shape=(config.batch_size, config.ndim_z / 2), dtype=tf.int32)
    class_fake = tf.zeros(shape=(config.batch_size, config.ndim_z / 2), dtype=tf.int32)
    loss_discriminator_z = opt.softmax_cross_entropy(D_z, D_z_prior,
                                                     class_fake,
                                                     class_true
                                                     )
    loss_encoder_z = opt.softmax_cross_entropy(D_z, D_z_prior, \
                                               class_fake,
                                               class_true,
                                               for_generator=True
                                               )
    loss_discriminator_img = opt.softmax_cross_entropy(D_img, D_img_prior,
                                                       class_fake,
                                                       class_true
                                                       )
    loss_decoder_img = opt.softmax_cross_entropy(D_img, D_img_prior,
                                                 class_fake,
                                                 class_true,
                                                 for_generator=True)
    loss_resconstruction = opt.euclidean_distance(images, reconstruction)

    # Variables Collection
    variables_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    variables_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    variables_discriminator_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_z')
    variables_discriminator_img = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_img')

    # Optimizer
    counter_encoder = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_resconstruction = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_discriminator_z = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_discriminator_img = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)

    opt_resconstruction = opt.optimize(loss_resconstruction, variables_decoder + variables_encoder,
                                       optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                       learning_rate=learning_rate, global_step=counter_resconstruction
                                       )

    opt_discriminator_z = opt.optimize(config.scale_ratio * loss_discriminator_z, variables_discriminator_z,
                                       optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                       learning_rate=learning_rate * 10, global_step=counter_discriminator_z
                                       )

    opt_discriminator_img = opt.optimize(loss_discriminator_img, variables_discriminator_img,
                                         optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                         learning_rate=learning_rate, global_step=counter_discriminator_img
                                         )

    opt_encoder_z = opt.optimize(config.scale_ratio * loss_encoder_z, variables_encoder,
                               optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                               learning_rate=learning_rate * 10, global_step=counter_encoder
                               )

    opt_decoder_img = opt.optimize(loss_decoder_img, variables_decoder,
                               optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                               learning_rate=learning_rate, global_step=counter_encoder
                               )

    # output what we want
    graph_handle = Object()
    graph_handle.x = images
    graph_handle.z = z_sampler
    graph_handle.x_s = img_sampler

    graph_handle.x_ = reconstruction
    graph_handle.z_r = z_representation

    graph_handle.opt_r = opt_resconstruction
    graph_handle.opt_dz = opt_discriminator_z
    graph_handle.opt_dimg = opt_discriminator_img
    graph_handle.opt_e = opt_encoder_z
    graph_handle.opt_d = opt_decoder_img
    graph_handle.loss_dz = loss_discriminator_z
    graph_handle.loss_dimg = loss_discriminator_img
    graph_handle.loss_e = loss_encoder_z
    graph_handle.loss_d = loss_decoder_img
    graph_handle.loss_r = loss_resconstruction
    graph_handle.lr = learning_rate

    return graph_handle
