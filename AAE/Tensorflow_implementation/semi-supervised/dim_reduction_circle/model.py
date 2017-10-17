from components import *

'''

build graph

'''


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
    # Transform the yz to z_representation
    representation_yz = encoder_yz_representation(img_y, img_z)
    # put z_representation to encoder
    with tf.variable_scope('decoder'):
        reconstruction = decoder_zy_x(representation_yz)

    if is_test:
        test_handle = Object()
        test_handle.x = images
        test_handle.y = img_y
        test_handle.z = img_z
        test_handle.yz = representation_yz
        test_handle.x_r = reconstruction
        return test_handle

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
    variables_transform = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='transform')

    # Optimizer
    counter_encoder = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_encoder_supervised = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_resconstruction = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_discriminator_y = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_discriminator_z = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
    counter_transform = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)

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
