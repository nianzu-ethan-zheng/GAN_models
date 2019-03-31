"""
Author : Nianzu Ethan Zheng
Datetime : 2018-2-2
Place: Shenyang China
Copyright
"""
from component_bn import *
import sequential.losses as Loss
from sequential.optimizer import optimize
from sequential.utils import logger


class Object:
    pass


def build_graph(is_test=False):
    logger.info("Set a placeholder")
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
    lr = tf.placeholder(
        dtype=tf.float32,
        shape=[]
    )
    # process
    pt = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, config.ndim_z]
    )
    pi_int = tf.placeholder(
        dtype=tf.float32,
        shape=[config.batch_size, 1, config.ndim_x, 1]
    )

    logger.info("A model is being built")
    # conditional AAE: B-> z -> B'
    z_img = encoder_x_z(
        img
    )
    img_z = decoder_z_x(
        img_cond,
        z_img,
        z_ept
    )
    # conditional Latent Regressor-GAN: z -> B' -> z'
    img_latent = decoder_z_x(
        img_cond,
        z_lat,
        z_ept,
        reuse=True
    )
    z_let = encoder_x_z(
        img_latent,
        reuse=True
    )

    # optimal_adjustment: r = r' + dr
    with tf.variable_scope('optimal_adjustment'):
        r, d = tf.split(
            img_latent,
            [config.ndim_r, config.ndim_d],
            axis=2
        )
        rd = tf.concat(
            [r,
             tf.zeros_like(d)],
            axis=2
        )
        pi = pi_int + rd

    # Process Validation: r -> Q
    po = process_x(pi)
    dq = po - pt

    # Test Phase
    if is_test:
        test_handle = Object()
        test_handle.z_e = z_ept
        test_handle.z_l = z_lat
        test_handle.z_img = z_img
        test_handle.x = img
        test_handle.x_c = img_cond
        test_handle.x_r = img_z
        test_handle.x_lat = img_latent
        test_handle.p_in = pi_int
        test_handle.p_i = pi
        test_handle.p_o = po
        test_handle.p_t = pt
        test_handle.dq = dq
        return test_handle

    # Discriminator on z
    D_z = discriminator_z(
        z_img
    )
    D_z_prior = discriminator_z(
        z_prior,
        reuse=True
    )

    # Discriminator on img
    D_img = discriminator_img(
        img_cond,
        img_latent,
        z_ept
    )
    D_img_prior = discriminator_img(
        img_cond,
        img_prior,
        z_ept,
        reuse=True
    )

    logger.info("The model has been built")

    logger.info("Start define loss function")
    class_true = tf.ones(
        shape=(config.batch_size, config.ndim_z / 2),
        dtype=tf.int32
    )
    class_fake = tf.zeros(
        shape=(config.batch_size, config.ndim_z / 2),
        dtype=tf.int32
    )
    loss_discriminator_z = Loss.softmax_cross_entropy(
        D_z,
        D_z_prior,
        class_fake,
        class_true
    )
    loss_encoder_z = Loss.softmax_cross_entropy(
        D_z,
        D_z_prior,
        class_fake,
        class_true,
        for_generator=True
    )
    loss_discriminator_img = Loss.softmax_cross_entropy(
        D_img,
        D_img_prior,
        class_fake,
        class_true
    )
    loss_decoder_img = Loss.softmax_cross_entropy(
        D_img,
        D_img_prior,
        class_fake,
        class_true,
        for_generator=True
    )
    logger.info('L2 latent loss function')
    loss_latent = Loss.euclidean_distance(
        z_lat,
        z_let
    )
    loss_r = Loss.euclidean_distance(
        img,
        img_z
    )
    # process
    loss_process = Loss.euclidean_distance(
        po,
        pt
    )
    # additional loss function
    loss_dq = Loss.euclidean_distance(
        z_ept,
        dq
    )
    loss_tv = Loss.euclidean_distance(
        r,
        tf.zeros_like(r)
    )
    logger.info('To sum up all the loss function')
    loss_EG = loss_r * config.coeff_rest + \
              loss_decoder_img + \
              loss_encoder_z * config.coeff_z + \
              loss_latent * config.coeff_lat
    loss_Dz = loss_discriminator_z
    loss_Di = loss_discriminator_img
    loss_GP = loss_dq + \
              loss_tv * config.coeff_tv
    loss_P = loss_process

    logger.info('Variables Collection')
    variables_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    variables_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    variables_discriminator_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_z')
    variables_discriminator_img = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_img')
    variables_process = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='process')

    [logger.info(var) for var in tf.trainable_variables()]
    logger.info('Optimizer')
    global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)

    opt_EG = optimize(
        loss_EG,
        variables_decoder + variables_encoder,
        global_step=global_step,
        learning_rate=lr
    )
    opt_Dz = optimize(
        loss_Dz,
        variables_discriminator_z,
        learning_rate=lr,
        global_step=None
    )
    opt_Di = optimize(
        loss_Di,
        variables_discriminator_img,
        learning_rate=lr,
        global_step=None
    )
    opt_P = optimize(
        loss_P,
        variables_process,
        learning_rate=lr,
        global_step=None
    )

    opt_GP = optimize(
        loss_GP,
        variables_decoder,  # variables_process + variables_decoder
        learning_rate=lr,
        global_step=None
    )

    # output what we want
    graph_handle = Object()
    graph_handle.x = img
    graph_handle.z_r = z_img
    graph_handle.z_p = z_prior
    graph_handle.z_l = z_lat
    graph_handle.z_e = z_ept
    graph_handle.x_c = img_cond
    graph_handle.x_s = img_prior
    graph_handle.x_ = img_z
    graph_handle.p_in = pi_int
    graph_handle.p_i = pi
    graph_handle.p_o = po
    graph_handle.p_ot = pt

    graph_handle.opt_r = opt_EG
    graph_handle.opt_dz = opt_Dz
    graph_handle.opt_dimg = opt_Di
    graph_handle.opt_p = opt_P
    graph_handle.opt_q = opt_GP

    graph_handle.loss_r = loss_r
    graph_handle.loss_e = loss_encoder_z
    graph_handle.loss_d = loss_decoder_img
    graph_handle.loss_l = loss_latent
    graph_handle.loss_eg = loss_EG

    graph_handle.loss_dz = loss_discriminator_z
    graph_handle.loss_dimg = loss_discriminator_img
    graph_handle.loss_p = loss_process

    graph_handle.loss_q = loss_dq
    graph_handle.loss_tv = loss_tv
    graph_handle.loss_gp = loss_GP
    graph_handle.lr = lr

    return graph_handle


if __name__ == '__main__':
    build_graph()
    # [logger.info(var) for var in variables_encoder]
    # [logger.info(var) for var in variables_decoder]
    # [logger.info(var) for var in variables_discriminator_z]
    # [logger.info(var) for var in variables_discriminator_img]
    # [logger.info(var) for var in variables_process]
    # logger.info(loss_EG)
