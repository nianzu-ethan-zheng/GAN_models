"""
Author : Nianzu Ethan Zheng
Datetime : 2018-2-2
Place: Shenyang China
Copyright
For Mutal information Metrics
"""
import os, sys
import tensorflow as tf
import numpy as np
from model import build_graph, config

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from data_providers.data_provider import SuvsDataProvider
from plot import Visualizer
from sequential.utils import pickle_save, pickle_load
from sampler import sampler_switch

vis = Visualizer()
dp = SuvsDataProvider(num_validation=config.num_vad, shuffle='every_epoch')
config.is_train = False
test_times = 60


def main(db='gs'):
    tf.reset_default_graph()
    config.batch_size = dp.valid.num_sample
    config.distribution_sampler = db
    with tf.device(config.device):
        t = build_graph(is_test=True)

    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=True
                                  )) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_path))

        q_errors = []
        r_adjs = []
        z_adjs = []

        z_true = sess.run(t.z_img, feed_dict={
            t.x: dp.test.rd
        })

        for time in range(test_times):
            z_latent = sampler_switch(config)
            q_error = sess.run(t.dq, feed_dict={
                t.z_e: dp.test.e,
                t.x_c: dp.test.c,
                t.z_l: z_latent,
                t.p_in: dp.test.rd,
                t.p_t: dp.test.q,
            })
            r_adj = sess.run(t.x_lat, feed_dict={
                t.x_c: dp.test.c,
                t.z_l: z_latent,
                t.z_e: dp.test.e,
            })
            z_adj = sess.run(t.z_img, feed_dict={
                t.x: r_adj
            })

            q_errors.append(q_error)
            r_adjs.append(r_adj)
            z_adjs.append(z_adj)

        q_errors = (np.array(q_errors) - np.expand_dims(dp.test.e, axis=0))**2
        r_adjs = np.array(r_adjs).reshape(-1, config.ndim_x)
        z_adjs = np.array(z_adjs).reshape(-1, config.ndim_z)

    # revise the number of batch size
    tf.reset_default_graph()
    config.batch_size = dp.train_l.num_sample

    with tf.device(config.device):
        t = build_graph(is_test=True)
    with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=True
                                  )) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_path))
        z_train = sess.run(t.z_img, feed_dict={
            t.x: dp.train.rd
        })


    pickle_save([q_errors, r_adjs, z_adjs, z_true,z_train],
                ["productions", "adjustment", "latent_variables"],
                '{}/{}-metric_plus3.pkl'.format(config.logs_path, config.get_description()))
    print('{}/{}-metric_plus3.pkl have been saved'.format(config.logs_path, config.get_description()))

if __name__ == "__main__":
    # main
    main('gs')
    main('gm')
    main('ud')
    main('us')
    main('sr')

    # file_path = '{}/{}-metric_plus2.pkl'.format(config.logs_path, config.description)
    # [q_errors, r_adjs, z_adjs, z_true, z_train], name = pickle_load(file_path)
    #                                                                      # Load data
    #
    # # latent space
    # vis.jointplot(z_adjs[:, 0], z_adjs[:, 1], config.latent_path, name='Scatter_latent_1')
    # vis.jointplot(z_true[:, 0], z_true[:, 1], config.latent_path, name='Scatter_latent_2')
    #
