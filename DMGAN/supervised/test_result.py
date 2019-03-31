"""
Author : Nianzu Ethan Zheng
Datetime : 2018-2-2
Place: Shenyang China
Copyright
For Visualization of training process and for Comparision of the different parameters
"""
import os, sys
import tensorflow as tf
from model import build_graph, config

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from data_providers.data_provider import SuvsDataProvider
from plot import Visualizer
from sequential.utils import pickle_save, copy_file, anti_norm, logger, pickle_load
from sampler import sampler_switch

vis = Visualizer()
dp = SuvsDataProvider(num_validation=config.num_vad, shuffle='every_epoch')
# set the flag config.is_train to False for batch_normalization
config.is_train = False
config.batch_size = dp.valid.num_sample


def main():
    with tf.device(config.device):
        t = build_graph(is_test=True)

    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
    )) as sess:

        logger.info(config.ckpt_path)
        saver = tf.train.Saver()
        saver.restore(sess,
                      tf.train.latest_checkpoint(config.ckpt_path))
        logger.info("Loading model completely")
        z_latent = sampler_switch(config)
        d_q = sess.run(t.p_o, feed_dict={
            t.z_e: dp.test.e,
            t.x_c: dp.test.c,
            t.z_l: z_latent,
            t.p_in: dp.test.rd,
        })
        r_p = sess.run(t.p_i, feed_dict={
            t.x_c: dp.test.c,
            t.z_l: z_latent,
            t.z_e: dp.test.e,
            t.p_in: dp.test.rd
        })

        # inverse the scaled output
        qm, qr, rdm, rdr = dp.out.qm, dp.out.qr, dp.out.rdm, dp.out.rdr
        actual_Q = anti_norm(dp.test.q, qm, qr)
        result_Q = anti_norm(d_q, qm, qr)
        actual_r = anti_norm(dp.test.rd, rdm, rdr)
        result_r = anti_norm(r_p, rdm, rdr)

        # save the result
        ensemble = {'actual_Q': actual_Q,
                    'result_Q': result_Q,
                    'actual_r': actual_r,
                    'result_r': result_r
                    }

        path = os.path.join(config.logs_path, config.description + '-test.pkl')
        pickle_save(
            ensemble,
            'test_result',
            path
        )
        copy_file(path, config.history_test_path)

        # visualize the process
        vis.cplot(actual_Q[:, 0],
                  result_Q[:, 0],
                  ['Q1', 'origin', 'modify'],
                  config.t_p
                  )
        vis.cplot(actual_Q[:, 1],
                  result_Q[:, 1],
                  ['Q2', 'origin', 'modify'],
                  config.t_p
                  )
        for num in range(6):
            vis.cplot(actual_r[:, num],
                      result_r[:, num],
                      ['R{}'.format(num+1), 'origin', 'modify'],
                      config.t_p)

if __name__ == "__main__":
    # test result
    main()

    # process loss
    path = os.path.join(
        config.logs_path, config.description + '-train.pkl'
    )

    logger.info("{}".format(path))
    hist_value, hist_head = pickle_load(path, use_pd=True)
    for loss_name in [
        'R_err',
        'GE_err',
        'EG_err',
        'GPt_err',
        'GP_err',
        'Pt_err',
        'P_err',
    ]:
        vis.tsplot(hist_value[loss_name], loss_name, config.loss_path)

    vis.dyplot(hist_value['Dz_err'],
               hist_value['Ez_err'], 'z',
               config.loss_path)
    vis.dyplot(hist_value['Di_err'],
               hist_value['Gi_err'], 'img',
               config.loss_path)



