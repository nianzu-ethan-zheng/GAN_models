##################################################################################
#                                 U-Net

##################################################################################
from sequential.block_provider import block_provider
from sequential.layers import max_pool, conv2d
import tensorflow as tf
from sequential.utils import log
from base_options import BaseConfig

config = BaseConfig()
bp = block_provider(mode=config.block_name)


@log
def down_block(_input, channels, is_training=None):
    return bp.downward_block(_input, channels, is_training=is_training)


@log
def up_block(_input, _input_skip, out_features, out_shape=None, is_training=None):
    return bp.upward_block(_input, _input_skip, out_features,
                           out_shape=out_shape, is_training=is_training)


class ru_net:
    def __init__(self):
        self.depth = 3
        self.drop = config.drop

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_number = 1
            for dim in shape:
                variable_number *= dim.value
            total_parameters += variable_number
        print('Total training params:%.1fM' % (total_parameters / 1e6))

    def mask_with_prob(self):
        """
        Get mask and corresponding prob

        Return  A mask , with shape 3 x level, and A scalar, prob
        """
        mask = tf.random_shuffle([0] + [1] * (3 * self.depth - 1))
        mask = tf.to_float(mask)
        unit = [i for i in range(1, self.depth + 1)]
        power = unit + [i for i in range(self.depth, 0, -1)] + unit
        prob = tf.reduce_sum((1 - tf.pow(0.5, power)) * (1.0 - mask))
        if not self.drop:
            mask = tf.to_float([1]*(3 * self.depth))
            prob = 1
        return mask, prob

    def build_ru_net(self, _input, is_training=True):
        with tf.variable_scope('Initial_convolution'):
            output = conv2d(_input, out_features=64, kernel_size=3)

        block_output = {}
        mask, prob = self.mask_with_prob()
        with tf.name_scope('encoder_section'):
            scope_name = 'block_1'
            u_input = output
            number_outputs = 64
            with tf.variable_scope(scope_name):
                output = down_block(u_input, number_outputs, is_training=is_training)
                block_output[scope_name] = output

            number_outputs = 64
            for i in range(self.depth):
                block_id = i + 2
                scope_name = 'block_%d' % block_id
                u_input = max_pool(output, k=2) * mask[2 * self.depth + i]
                number_outputs *= 2
                with tf.variable_scope(scope_name):
                    output = down_block(u_input, number_outputs, is_training=is_training)
                    block_output[scope_name] = output

        with tf.name_scope('decoder_section'):
            number_outputs = 512
            for i in range(self.depth):
                block_id = i + self.depth + 2
                scope_name = 'block_%d' % block_id
                u_input = output * mask[self.depth + i]
                number_outputs /= 2
                with tf.variable_scope(scope_name):
                    if i != self.depth - 1:
                        output = up_block(u_input, block_output['block_%d' % (2 * self.depth + 2 - block_id)] * mask[i],
                                          number_outputs, is_training=is_training)
                    else:
                        output = up_block(u_input, block_output['block_%d' % (2 * self.depth + 2 - block_id)] * mask[i],
                                          number_outputs, out_shape=[1, 15], is_training=is_training)
                    block_output[scope_name] = output

        scope_name = 'transition_to_output'
        with tf.variable_scope(scope_name):
            output = conv2d(output, out_features=1, kernel_size=1)
            output = tf.nn.tanh(output, name='activation')
        return output / prob

