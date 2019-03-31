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
import matplotlib.pylab as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from data_providers.data_provider import SuvsDataProvider
from plot import Visualizer, scatter_labeled_z
from sequential.utils import pickle_save, pickle_load, logger
from sequential.metric import mi, cross_validate_sigma
from sampler import sampler_switch

vis = Visualizer()
dp = SuvsDataProvider(num_validation=config.num_vad, shuffle='every_epoch')
config.is_train = False
config.batch_size = dp.valid.num_sample
test_times = 60


def main():
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

        pickle_save([q_errors, r_adjs, z_adjs], ["productions", "adjustment", "latent_variables"],
                    '{}/{}-metric.pkl'.format(config.logs_path, config.description))

if __name__ == "__main__":
    # main
    main()
    # Save data
    file_path = '{}/{}-metric.pkl'.format(
        config.logs_path, config.description
    )
    # Load data
    [q_errors, r_adjs, z_adjs], name = pickle_load(
        file_path
    )

    #  MSE evaluate
    q_errors_ = np.sum(q_errors, axis=-1)  # * (np.square(np.array(dp.out.qr)).reshape(1, 1, config.ndim_y))
    mse = np.mean(q_errors_, axis=-1)
    mean_mse = np.mean(mse)
    std_mse = np.std(mse)
    logger.info("{}- MSE - mean and std: {:0.4f}+/- {:0.4f}".format(
        config.description, mean_mse, std_mse))

    # Energy evaluate
    r = np.reshape(r_adjs, [test_times, config.batch_size, -1])[:, :, :config.ndim_r]
    eng = np.mean(np.sum(np.square(r), axis=-1), axis=-1)
    mean_eng = np.mean(eng)
    std_eng = np.std(eng)
    logger.info("{}- Eng - mean and std: {:0.4f}+/- {:0.4f}".format(
        config.description, mean_eng, std_eng))

    # MI evaluate
    train_set = dp.out.eg
    train_set = np.squeeze(train_set, axis=[1, 3])[:, :config.ndim_r]
    test_set = r_adjs[:, :config.ndim_r]
    sigmas = np.linspace(1e-4, 2, 100)
    sigma, vals = cross_validate_sigma(
        train_set,
        test_set,
        sigmas,
        batch_size=59
    )
    # Visualize the likelihoods
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.plot(sigmas, vals, '-*r')
    ax.set(xlabel='$\sigma$', ylabel='Likelihood')
    yd, yu = ax.get_ylim()
    ax.text(1.5, yd * 0.8 + yu * 0.2, "$best \sigma = {:0.4f}$".format(sigma))
    fig.savefig(config.latent_path + '/Industry_window_width.png')
    plt.close()

    d, s, arg_h, H = mi(train_set, test_set, prior=[0.5, 0.5], sigma=sigma, batch_size=len(train_set))
    logger.info('{}-MI distance:{:0.4f}+/- {:0.4f}'.format(config.get_description(),d, s))

    # latent space
    label = np.tile(np.array(dp.test.e), [test_times, 1])
    scatter_labeled_z(z_adjs, label,
                      dir=config.latent_path,
                      filename='latent-Q2')
    vis.kdeplot(z_adjs[:, 0], z_adjs[:, 1], config.latent_path, name='Kde_Latent_Space')
    vis.jointplot(z_adjs[:, 0], z_adjs[:, 1], config.latent_path, name='Scatter_latent')
    #
