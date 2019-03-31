"""
Author : Nianzu Ethan Zheng
Datetime : 2018-2-2
Place: Shenyang China
Copyright
"""
from sequential.layers import conv2d, deconv2d, batch_norm, dropout
from sequential.utils import logger
from sequential.mixer import mixer
from base_options import BaseConfig
import tensorflow as tf

config = BaseConfig()
if (config.mixer is "gauss") and config.block_name not in ['vgg', 'dense', 'fra', 'fra_ts']:
    config.block_mode = 'bc'
    logger.info('The res and fra+ block mode has been revised: plain -> bc')
    

def block_provider(mode='dense'):
    if mode is "vgg":
        bp = VGG(
            layers_per_block=config.block_layers
        )
    elif mode is "res":
        bp = Res(
            layer_per_function=config.block_layers,  # default 2
            block_mode=config.block_mode,  # default plain
        )
    elif mode is "dense":
        bp = Den(
            layers_per_block=config.block_layers,  # default 4
            block_mode=config.block_mode,  # default plain
            growth_rate=None,  # default None
        )
    elif mode is "fra":
        bp = Fra(
            num_columns=config.block_layers  # default 3
        )
    elif mode is "fra_u":
        bp = Fra_U(
            num_columns=config.block_layers,  # default 3
            block_mode=config.block_mode  # default bc
        )
    elif mode is "fra_ts":
        bp = Fra_TS(
            num_columns=config.block_layers, # default 3
        )
    elif mode is 'inception':
        bp = Inception(
            block_mode=config.block_mode
        )
    else:
        raise Exception('please set a valid block name')
    return bp


def composite_function(_input, out_features, kernel_size=3,
                       is_training=None, keep_prob=1):
    with tf.variable_scope('composite_function', reuse=False):
        output = conv2d(_input, out_features=out_features, kernel_size=kernel_size)
        output = batch_norm(output, is_training=is_training)
        output = tf.nn.relu(output)
        output = dropout(output, keep_prob=keep_prob, is_training=is_training)
    return output


class Network:
    def add_block(self, _input, out_features, is_training=None):
        raise NotImplementedError

    def add_bottleneck(self, _input, out_features, is_training):
        with tf.variable_scope('convert_feature_maps'):
            out = composite_function(_input, out_features, kernel_size=1, is_training=is_training)
        return out

    def downward_block(self, _input, out_features, is_training=None):
        with tf.variable_scope('down_block'):
            out = self.add_block(_input, out_features, is_training=is_training)
        return out

    def upward_block(self, _input, _input_skip, out_features, out_shape=None, is_training=None):
        with tf.variable_scope('up_block'):
            _input = deconv2d(_input, out_shape=out_shape)
            _input = mixer(_input, _input_skip, mode=config.mixer)
            out = self.add_block(_input, out_features, is_training=is_training)
        return out


class VGG(Network):
    def __init__(self, layers_per_block=2):
        self.layers_per_block = layers_per_block

    def add_block(self, _input, out_features, is_training=None):
        out = _input
        for layer in range(self.layers_per_block):
            with tf.variable_scope('layer_%d' % layer):
                out = composite_function(out, out_features, is_training=is_training)
        return out


class Res(Network):
    def __init__(self, layer_per_function=2, block_mode='bc'):
        self.layer_per_function = layer_per_function
        self.block_mode = block_mode

    def add_block(self, _input, out_features, is_training=None, bc=False):
        output = _input
        _out_features = _input.get_shape().as_list()[-1]
        for layer in range(self.layer_per_function):
            with tf.variable_scope('layer_%d' % layer):
                output = composite_function(output, _out_features, is_training=is_training)
        output = tf.add(_input, output, name='res_operation')
        if self.block_mode is 'bc':
            with tf.variable_scope('convert_feature_maps'):
                output = self.add_bottleneck(output, out_features, is_training=is_training)
        return output


class Inception(Network):
    def __init__(self, block_mode='bc'):
        self.block_mode = block_mode

    def add_internal_layer(self, _input, growth_rate, kernel_size=3,is_training=None):
        comp_out = composite_function(_input, growth_rate,
                                      kernel_size=kernel_size,
                                      is_training=is_training)
        output = tf.concat(axis=3, values=[_input, comp_out])
        return output

    def add_block(self, _input, out_features, is_training=None, bc=False):
        ks = [1, 3, 5]
        growth_rate = []
        if self.block_mode is 'bc':
            growth_rate = out_features
        if self.block_mode is 'plain':
            growth_rate = int(out_features / len(ks))
        feature_maps = []
        for layer, k in enumerate(ks):
            with tf.variable_scope('Layer_%d' % layer):
                out = self.add_internal_layer(_input, growth_rate=growth_rate,
                                              kernel_size=k, is_training=is_training)
            feature_maps.append(out)
        output = tf.concat(feature_maps, axis=-1)
        with tf.variable_scope('convert_feature_maps'):
            output = self.add_bottleneck(output, out_features, is_training=is_training)
        return output

class Den(Network):
    def __init__(self, layers_per_block=2, block_mode='bc', growth_rate=None):
        self.layers_per_block = layers_per_block
        self.block_mode = block_mode
        self.growth_rate = growth_rate

    def add_internal_layer(self, _input, growth_rate, is_training=None):
        comp_out = composite_function(_input, growth_rate, is_training=is_training)
        output = tf.concat(axis=3, values=[_input, comp_out])
        return output

    def add_block(self, _input, out_features, is_training=None):
        output = _input
        growth_rate = []
        if self.growth_rate is None:
            if self.block_mode is 'bc':
                growth_rate = out_features
            if self.block_mode is 'plain':
                growth_rate = int(out_features / self.layers_per_block)
        else:
            growth_rate = self.growth_rate

        for layer in range(self.layers_per_block):
            with tf.variable_scope('Layer_%d' % layer):
                output = self.add_internal_layer(output, growth_rate, is_training=is_training)
        with tf.variable_scope('convert_feature_maps'):
            output = self.add_bottleneck(output, out_features, is_training=is_training)
        return output


class Fra(Network):
    def __init__(self, num_columns=2):
        self.num_columns = num_columns # total number: 2^C-1; depth: 2^(C-1)

    def tensor_shape(self, tensor):
        return tensor.get_shape().as_list()

    def join(self, columns):
        if len(columns) == 1:
            return columns[0]
        with tf.variable_scope('Join'):
            columns = tf.convert_to_tensor(columns)
            out = tf.reduce_mean(columns, 0)
        return out

    def fractal_block(self, _input,
                      num_columns,
                      out_features,
                      is_training=True,
                      reuse=False):
        join = self.join

        def fractal_expand(__input, _num_columns):
            if _num_columns == 1:
                out = join([composite_function(__input, out_features, is_training=is_training)])
                return out
            with tf.variable_scope('columns_left_%d' % _num_columns):
                left = composite_function(__input, out_features, is_training=is_training)
            with tf.variable_scope('columns_right_%d' % _num_columns):
                with tf.variable_scope('front_part'):
                    right = fractal_expand(__input, _num_columns - 1)
                with tf.variable_scope('back_part'):
                    right = fractal_expand(right, _num_columns - 1)
            columns = [left] + [right]
            return join(columns)

        with tf.variable_scope('fractal', reuse=reuse):
            net = fractal_expand(_input, num_columns)
            return net

    def add_block(self, _input, out_features, is_training=None):
        return self.fractal_block(_input, num_columns=self.num_columns, out_features=out_features, is_training=is_training)


class Fra_TS(Network):
    def __init__(self, num_columns=2):
        self.num_columns = num_columns # total number: 2^C-1; depth: 2^(C-1)

    def tensor_shape(self, tensor):
        return tensor.get_shape().as_list()

    def apply_mask(self, mask, columns):
        """Use a boolean mask to zero out some columns"""
        tensor = tf.convert_to_tensor(columns)
        mask = tf.cast(mask, tensor.dtype)  # broadcasting required
        return tf.transpose(tf.multiply(tf.transpose(tensor), mask))

    def rand_column(self, columns):
        """Zeros out all expect on of columns

        Used for rounds with global drop path

        """
        num_columns = self.tensor_shape(columns)[0]
        mask = tf.random_shuffle([True] + [False] * (num_columns - 1))
        return self.apply_mask(mask, columns) * num_columns

    def drop_some(self, columns, drop_prob=0.15):
        """Zero out columns with probability 'drop_prop"

        Used for rounds of local drop path
        """
        num_columns = self.tensor_shape(columns)[0]
        mask = tf.random_uniform([num_columns]) > drop_prob
        scale = num_columns / tf.reduce_sum(tf.cast(mask, tf.float32))
        return tf.cond(tf.reduce_any(mask),
                       lambda: self.apply_mask(mask, columns) * scale,
                       lambda: self.rand_column(columns))

    def coin_flip(self, prob=0.5):
        """Random boolean variable, with "Prob"
        Used to choose between local and global drop path
        """
        with tf.variable_scope('CoinFlip'):
            out = tf.random_uniform([1])[0] > prob
        return out

    def drop_path(self, columns, coin):
        with tf.variable_scope('drop_path'):
            out = tf.cond(coin,
                          lambda: self.drop_some(columns),
                          lambda: self.rand_column(columns))
        return out

    def join(self, columns, coin, is_training):
        if len(columns) == 1:
            return columns[0]
        with tf.variable_scope('Join'):
            columns = tf.convert_to_tensor(columns)
            columns = tf.cond(pred=tf.cast(is_training, dtype=tf.bool),
                              fn1=lambda: self.drop_path(columns, coin),
                              fn2=lambda: columns)
        out = tf.reduce_mean(columns, 0)
        return out


    def fractal_block(self, _input,
                      num_columns,
                      out_features,
                      is_training=True,
                      joined=True,
                      reuse=False):
        join = self.join

        def fractal_expand(x, _num_columns, _joined):
            adjoin = lambda cols: join(cols, coin, is_training=is_training) if _joined else cols
            if _num_columns == 1:
                out = adjoin([composite_function(x, out_features, is_training=is_training)])
                return out
            with tf.variable_scope('columns_left_%d' % _num_columns):
                left = composite_function(x, out_features, is_training=is_training)
            with tf.variable_scope('columns_right_%d' % _num_columns):
                with tf.variable_scope('front_part'):
                    right = fractal_expand(x, _num_columns - 1, _joined=_joined)
                with tf.variable_scope('back_part'):
                    right = fractal_expand(right, _num_columns - 1, _joined=_joined)
            columns = [left] + [right]
            return adjoin(columns)

        with tf.variable_scope('fractal', reuse=reuse):
            coin = self.coin_flip()
            net = fractal_expand(_input, num_columns, _joined=joined)
            return net

    def add_block(self, _input, out_features, is_training=None):
        return self.fractal_block(_input, out_features=out_features, num_columns=self.num_columns, is_training=is_training)


class Fra_U(Network):
    def __init__(self, num_columns=2, block_mode='bc'):
        self.num_columns = num_columns # total number: 2^C-1; depth: 2^(C-1)
        self.block_mode = block_mode

    def tensor_shape(self, tensor):
        return tensor.get_shape().as_list()

    def join(self, columns):
        if len(columns) == 1:
            return columns[0]
        with tf.variable_scope('Join'):
            num_split = self.tensor_shape(columns[1])[-1]/self.tensor_shape(columns[0])[-1]
            columns = [columns[0]] + tf.split(columns[1], int(num_split), axis=3)
            columns = tf.convert_to_tensor(columns)
            shape = self.tensor_shape(columns)
        out = tf.reshape(columns, shape=[shape[1], shape[2], shape[3], -1])
        return out

    def fractal_block(self, _input,
                      num_columns,
                      out_features,
                      is_training=True,
                      joined=True,
                      reuse=False):
        join = self.join

        def fractal_expand(x, _num_columns, _joined):
            adjoin = lambda cols: join(cols) if _joined else cols
            if _num_columns == 1:
                out = adjoin([composite_function(x, out_features, is_training=is_training)])
                return out
            with tf.variable_scope('columns_left_%d' % _num_columns):
                left = composite_function(x, out_features, is_training=is_training)
            with tf.variable_scope('columns_right_%d' % _num_columns):
                with tf.variable_scope('front_part'):
                    right = fractal_expand(x, _num_columns - 1, _joined=_joined)
                with tf.variable_scope('back_part'):
                    right = fractal_expand(right, _num_columns - 1, _joined=_joined)
            columns = [left] + [right]
            return adjoin(columns)

        with tf.variable_scope('fractal', reuse=reuse):
            net = fractal_expand(_input, num_columns, _joined=joined)
            return net

    def add_block(self, _input, out_features, is_training=None):
        growth_rate = []
        if self.block_mode is 'plain':
            growth_rate = out_features // self.num_columns
        if self.block_mode is 'bc':
            growth_rate = out_features
        output = self.fractal_block(_input, out_features=growth_rate, num_columns=self.num_columns, is_training=is_training)
        with tf.variable_scope('convert_feature_maps'):
            output = self.add_bottleneck(output, out_features, is_training=is_training)
        return output
