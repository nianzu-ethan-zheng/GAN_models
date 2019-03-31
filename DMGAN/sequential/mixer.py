"""
Author : Nianzu Ethan Zheng
Datetime : 2018-2-2
Place: Shenyang China
Copyright
"""
import tensorflow as tf


def mixer(_input, _input_skip, mode='gauss'):
    """Mix up the input from higher level and lateral input
    """

    def gauss_mixer(u, z):
        """u, z equal to  _input, _input_skip
        """
        tensor_shape = lambda tensor: tensor.get_shape().as_list()
        if tensor_shape(u) != tensor_shape(z):
            raise Exception('the shape of input does no match the shape of lateral input')
        shape = tensor_shape(u)
        shape[0] = 1
        # wi = lambda init_value, name: tf.Variable(init_value * tf.ones(shape), name=name)
        wi = lambda num,name: tf.get_variable(name=name,
                                              shape=shape,
                                              initializer=tf.contrib.layers.variance_scaling_initializer())
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')
        mul = tf.multiply
        mu = mul(a1, tf.sigmoid(mul(a2, u) + a3)) + mul(a4, u) + a5
        v = mul(a6, tf.sigmoid(mul(a7, u) + a8)) + mul(a9, u) + a10

        out = mul((z - mu), v) + mu
        return out

    def general_gauss_mixer(u, z):
        """u, z equal to  _input, _input_skip
        """
        tensor_shape = lambda tensor: tensor.get_shape().as_list()
        if tensor_shape(u) != tensor_shape(z):
            raise Exception('the shape of input does no match the shape of lateral input')
        shape = tensor_shape(u)
        shape[0] = 1
        # wi = lambda init_value, name: tf.Variable(init_value * tf.ones(shape), name=name)
        wi = lambda num,name: tf.get_variable(name=name,
                                              shape=shape,
                                              initializer=tf.contrib.layers.variance_scaling_initializer())
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')
        mul = tf.multiply
        mu = mul(a1, tf.sigmoid(mul(a2, u) + a3)) + mul(a4, u) + a5
        v = mul(a6, tf.sigmoid(mul(a7, u) + a8)) + mul(a9, u) + a10

        out = mul((z - mu), v) + mu
        return tf.concat([_input, _input_skip, out], axis=-1)


    def linear_mixer(u, z):
        """u, z equal to  _input, _input_skip
        """
        tensor_shape = lambda tensor: tensor.get_shape().as_list()
        if tensor_shape(u) != tensor_shape(z):
            raise Exception('the shape of input does no match the shape of lateral input')
        shape = tensor_shape(u)
        shape[0] = 1
        # wi = lambda init_value, name: tf.Variable(init_value * tf.ones(shape), name=name)
        wr = lambda name: tf.get_variable(name=name,
                                          shape=shape,
                                          initializer=tf.contrib.layers.variance_scaling_initializer())
        a1 = wr('a1')
        a2 = wr('a2')
        mul = tf.multiply
        return mul(a1, u) + mul(a2, z)

    def general(u, z):
        tensor_shape = lambda tensor: tensor.get_shape().as_list()
        if tensor_shape(u) != tensor_shape(z):
            raise Exception('the shape of input does no match the shape of lateral input')
        shape = tensor_shape(u)
        shape[0] = 1
        # wi = lambda init_value, name: tf.Variable(init_value * tf.ones(shape), name=name)
        wr = lambda name, shape: tf.get_variable(name=name,
                                                 shape=shape,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer())
        a1 = wr('a1', shape)
        a2 = wr('a2', shape)
        mul = tf.multiply
        lin = mul(a1, u) + mul(a2, z)
        return tf.concat([_input, _input_skip, lin], axis=-1)

    with tf.variable_scope('combinator'):
        if mode == 'concat':
            return tf.concat([_input, _input_skip], axis=-1)
        elif mode == 'gauss':
            return gauss_mixer(_input, _input_skip)
        elif mode == 'res':
            return _input + _input_skip
        elif mode == 'lin':
            return linear_mixer(_input, _input_skip)
        elif mode == 'gen':
            return general(_input, _input_skip)
        elif mode == 'ggs':
            return general_gauss_mixer(_input, _input_skip)



