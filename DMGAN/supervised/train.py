# Standard library
import tensorflow as tf
import numpy as np

import os, sys
# Packages
from model import build_graph, config

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from base_options import BaseConfig
from data_providers.data_provider import SuvsDataProvider
from process import Process
import sampler
from sequential.learning_rate import create_lr_schedule
from sequential.utils import pickle_save, pickle_load, copy_file, logger


def main(run_load_from_file=False):
    config = BaseConfig()
    config.folder_init()
    dp = SuvsDataProvider(num_validation=config.num_vad, shuffle='every_epoch')
    max_epoch = 500
    batch_size_l = config.batch_size
    path = os.path.join(config.logs_path, config.description + '-train.pkl')

    # training
    with tf.device(config.device):
        h = build_graph()

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    saver = tf.train.Saver(
        max_to_keep=2
    )

    with tf.Session(config=sess_config) as sess:
        '''
         Load from checkpoint or start a new session

        '''
        if run_load_from_file:
            saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_path))
            training_epoch_loss, _ = pickle_load(path)
        else:
            sess.run(tf.global_variables_initializer())
            training_epoch_loss = []

        # Recording loss per epoch
        process = Process()
        lr_schedule = create_lr_schedule(lr_base=2e-4, decay_rate=0.1, decay_epochs=500,
                                         truncated_epoch=2000, mode=config.lr_schedule)
        for epoch in range(max_epoch):
            process.start_epoch()

            '''
            Learning rate generator

            '''
            learning_rate = lr_schedule(epoch)
            # Recording loss per iteration
            training_iteration_loss = []
            sum_loss_rest = 0
            sum_loss_dcm = 0
            sum_loss_gen = 0

            process_iteration = Process()
            data_size = dp.train_l.num_sample
            num_batch = data_size // config.batch_size
            for i in range(num_batch + 1):
                process_iteration.start_epoch()
                # Inputs
                # sample from data distribution
                batch_l = dp.train_l.next_batch(batch_size_l)
                z_prior = sampler.sampler_switch(config)
                # adversarial phase for discriminator_z
                _, Dz_err = sess.run([h.opt_dz, h.loss_dz], feed_dict={
                    h.x: batch_l.x,
                    h.z_p: z_prior,
                    h.lr: learning_rate,
                })
                z_latent = sampler.sampler_switch(config)
                _, Di_err = sess.run([h.opt_dimg, h.loss_dimg], feed_dict={
                    h.x_c: batch_l.c,
                    h.z_l: z_latent,
                    h.z_e: batch_l.e,
                    h.x_s: batch_l.x,
                    h.lr: learning_rate,
                })
                z_latent = sampler.sampler_switch(config)
                # reconstruction_phase
                _, R_err, Ez_err, Gi_err, GE_err, EG_err = sess.run(
                    fetches=[
                        h.opt_r,
                        h.loss_r,
                        h.loss_e,
                        h.loss_d,
                        h.loss_l,
                        h.loss_eg
                    ],
                    feed_dict={
                        h.x: batch_l.x,
                        h.z_p: z_prior,
                        h.x_c: batch_l.c,
                        h.z_l: z_latent,
                        h.z_e: batch_l.e,
                        h.x_s: batch_l.x,
                        h.lr: learning_rate,
                    })
                # process phase
                _, P_err = sess.run([h.opt_p, h.loss_p], feed_dict={
                    h.p_i: batch_l.rd,
                    h.p_ot: batch_l.q,
                    h.lr: learning_rate
                })
                # push process to normal
                z_latent = sampler.sampler_switch(config)
                _, GP_err = sess.run([h.opt_q, h.loss_q], feed_dict={
                    h.x_c: batch_l.c,
                    h.z_l: z_latent,
                    h.z_e: batch_l.e,
                    h.p_in: batch_l.rd,
                    h.p_ot: batch_l.q,
                    h.lr: learning_rate,
                })
                # recording loss function
                training_iteration_loss.append([
                    R_err, Ez_err, Gi_err, GE_err, EG_err, Dz_err, Di_err, P_err, GP_err
                ])
                sum_loss_rest += R_err
                sum_loss_dcm += Dz_err + Di_err
                sum_loss_gen += Gi_err + Ez_err

                if i % 10 == 0 and False:
                    process_iteration.display_current_results(i, num_batch, {
                        'reconstruction': sum_loss_rest / (i + 1),
                        'discriminator': sum_loss_dcm / (i + 1),
                        'generator': sum_loss_gen / (i + 1),
                    })

            # In end of epoch, summary the loss
            average_loss_per_epoch = np.mean(np.array(training_iteration_loss), axis=0)

            # validation phase
            num_test = dp.valid.num_sample // config.batch_size
            testing_iteration_loss = []
            for batch in range(num_test):
                z_latent = sampler.sampler_switch(config)
                batch_v = dp.valid.next_batch(config.batch_size)
                GPt_err = sess.run(h.loss_q, feed_dict={
                    h.x_c: batch_v.c,
                    h.z_l: z_latent,
                    h.z_e: batch_v.e,
                    h.p_in: batch_v.rd,
                    h.p_ot: batch_v.q,
                })
                Pt_err = sess.run(h.loss_p, feed_dict={
                    h.p_i: batch_v.rd,
                    h.p_ot: batch_v.q,
                })
                testing_iteration_loss.append([GPt_err, Pt_err])
            average_test_loss = np.mean(np.array(testing_iteration_loss), axis=0)

            average_per_epoch = np.concatenate(
                (average_loss_per_epoch, average_test_loss), axis=0)
            training_epoch_loss.append(average_per_epoch)

            # training loss name
            training_loss_name = [
                'R_err',
                'Ez_err',
                'Gi_err',
                'GE_err',
                'EG_err',
                'Dz_err',
                'Di_err',
                'P_err',
                'GP_err',
                'GPt_err',
                'Pt_err',
            ]

            if epoch % 10 == 0:
                process.format_meter(epoch, max_epoch, {
                    'R_err': average_per_epoch[0],
                    'Ez_err': average_per_epoch[1],
                    'Gi_err': average_per_epoch[2],
                    'GE_err': average_per_epoch[3],
                    'EG_err': average_per_epoch[4],
                    'Dz_err': average_per_epoch[5],
                    'Di_err': average_per_epoch[6],
                    'P_err': average_per_epoch[7],
                    'GP_err': average_per_epoch[8],
                    'GPt_err': average_per_epoch[9],
                    'Pt_err': average_per_epoch[10],
                })

            if (epoch % 1000 == 0 or epoch == max_epoch - 1) and epoch != 0:
                saver.save(sess, os.path.join(config.ckpt_path, 'model_checkpoint'), global_step=epoch)
                pickle_save(training_epoch_loss, training_loss_name, path)
                copy_file(path, config.history_train_path)


if __name__ == '__main__':
    main(run_load_from_file=False)
