import tensorflow as tf
from components import config, opt
import os, sys
import numpy as np
import pylab
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dataset


g_start = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_y])
g_ending = tf.placeholder(dtype=tf.float32, shape=[None, config.ndim_y])


def transform(y, reuse=False):
    with tf.variable_scope('transform') as scope:
        if reuse:
            scope.reuse_variables()
        img = tf.layers.dense(y, config.ndim_reduction, activation=None, use_bias=False,
                              kernel_initializer=tf.random_normal_initializer(0, 1))
        return img


starting_vector = transform(g_start)
ending_vector = transform(g_ending, reuse=True)

euclidean_distance = tf.reduce_sum(tf.square(starting_vector - ending_vector), axis=1)
euclidean_distance = tf.minimum(euclidean_distance, tf.fill(
    dims=tf.shape(euclidean_distance), value=config.cluster_head_distance_threshold
))
loss_transform = -tf.reduce_sum(euclidean_distance) / 2

counter_transform = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)

variables_transform = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='transform')

opt_transform = opt.optimize(loss_transform, variables_transform,
                                 optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
                                 learning_rate=0.1, global_step=counter_transform, is_clipping=True
                                 )


starting_labels, ending_labels = dataset.cluster_create_dataset()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(variables_transform)

for i in range(1500):
    sess.run([opt_transform,loss_transform], feed_dict={
        g_start: starting_labels,
        g_ending: ending_labels})

all_y = np.identity(10)
z = sess.run(starting_vector, feed_dict={g_start: all_y})
pylab.scatter(z[:,0],z[:,1])

import math
r = 3
Wc = np.zeros([config.ndim_y, config.ndim_reduction])
for m in range(config.ndim_y):
    Wc[m] = [r*math.cos(np.pi*2/config.ndim_y*m), r* math.sin(np.pi*2/config.ndim_y*m)]
z = np.matmul(all_y, Wc)


