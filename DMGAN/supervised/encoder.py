import tensorflow as tf
from sequential.layers import conv2d, max_pool, flatten, mlp
from sequential.block_provider import Res
from sequential.utils import logger


def encoder(_input, is_training, latent='gs'):
    res = Res(layer_per_function=2)
    with tf.variable_scope('initial_convolution'):
        out = conv2d(_input, out_features=64, kernel_size=3)
    with tf.variable_scope('res_block_1'):
        out = res.add_block(out, out_features=64, is_training=is_training, bc=False)
    with tf.variable_scope('down_sampling'):
        out = max_pool(out, k=2)
        out = conv2d(out, out_features=128, kernel_size=3)
    with tf.variable_scope('res_block_2'):
        out = res.add_block(out, out_features=128, is_training=is_training, bc=False)
    out = max_pool(out, k=2)
    out = flatten(out)
    out = mlp(out, 256, 'leaky', is_training, norm=True, name='mlp_1')
    if latent is 'dm':
        out = mlp(out, 2, None, is_training, norm=False, name='mlp_2')
    elif latent is 'gs':
        mu = mlp(out, 2, None, is_training, norm=False, name='mlp_2')
        sg = mlp(out, 2, None, is_training, norm=False, name='mlp_3')
        es = tf.random_normal(shape=tf.shape(mu), mean=0, stddev=0.3)
        out = mu + es * sg
    else:
        raise Exception('please input a valid mode')
    return out

if __name__ == "__main__":
    def log(out):
        logger.info(out.get_shape())

    img = tf.placeholder(
        dtype=tf.float32,
        shape=[32, 1, 15, 1]
    )
    logger.info(img.get_shape())
    logger.info(type(img))
    logger.info('{}'.format(isinstance(img, tf.Tensor)))
    # encoder(img, is_training=True, latent='gs')

    # [logger.info(var) for var in tf.trainable_variables()]
    # f= flatten(img)
    # log(f)
    # import numpy as np
    # print(np.prod(img.get_shape().as_list()[1:]))
    # log(tf.reshape(img, [-1, np.prod(img.get_shape().as_list()[1:])]))
    # logger.info(f.get_shape())

