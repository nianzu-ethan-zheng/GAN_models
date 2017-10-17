# Standard library
import tensorflow as tf
import numpy as np
import _pickle as pickle
import os, sys
# Packages
from model import build_graph, config

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from aae import Operation
import dataset
from process import Process
import sampler
import plot as plt


def main(run_load_from_file=False):
    # load MNIST images
    images, labels = dataset.load_test_images()

    # config
    opt = Operation()
    opt.check_dir(config.ckpt_dir, is_restart=False)
    opt.check_dir(config.log_dir, is_restart=True)

    max_epoch = 510
    num_trains_per_epoch = 500
    batch_size_u = 100

    # training
    with tf.device(config.device):
        h = build_graph()

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session(config=sess_config) as sess:
        '''
         Load from checkpoint or start a new session

        '''
        if run_load_from_file:
            saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_dir))
            training_epoch_loss, _ = pickle.load(open(config.ckpt_dir + '/pickle.pkl', 'rb'))
        else:
            sess.run(tf.global_variables_initializer())
            training_epoch_loss = []

        # Recording loss per epoch
        process = Process()
        for epoch in range(max_epoch):
            process.start_epoch(epoch, max_epoch)

            '''
            Learning rate generator

            '''
            learning_rate = 0.0001

            # Recording loss per iteration
            sum_loss_reconstruction = 0
            sum_loss_discrminator_z = 0
            sum_loss_discrminator_img = 0
            sum_loss_generator_z = 0
            sum_loss_generator_img = 0
            process_iteration = Process()
            for i in range(num_trains_per_epoch):
                process_iteration.start_epoch(i, num_trains_per_epoch)
                # Inputs
                '''
                _l -> labeled
                _u -> unlabeled

                '''
                images_u = dataset.sample_unlabeled_data(images, batch_size_u)
                if config.distribution_sampler == 'swiss_roll':
                    z_true_u = sampler.swiss_roll(batch_size_u,
                                                  config.ndim_z,
                                                  config.num_types_of_label
                                                  )
                elif config.distribution_sampler == 'gaussian_mixture':
                    z_true_u = sampler.gaussian_mixture(batch_size_u,
                                                        config.ndim_z,
                                                        config.num_types_of_label
                                                        )
                elif config.distribution_sampler == 'uniform_desk':
                    z_true_u = sampler.uniform_desk(batch_size_u,
                                                    config.ndim_z,
                                                    radius=2
                                                    )
                elif config.distribution_sampler == 'gaussian':
                    z_true_u = sampler.gaussian(batch_size_u,
                                                config.ndim_z,
                                                var=1
                                                )
                elif config.distribution_sampler == 'uniform':
                    z_true_u = sampler.uniform(batch_size_u,
                                               config.ndim_z,
                                               minv=-1,
                                               maxv=1
                                               )

                # reconstruction_phase
                _, loss_reconstruction = sess.run([h.opt_r, h.loss_r], feed_dict={
                    h.x: images_u,
                    h.lr: learning_rate
                })

                # adversarial phase for discriminator_z
                images_u_s = dataset.sample_unlabeled_data(images, batch_size_u)
                _, loss_discriminator_z = sess.run([h.opt_dz, h.loss_dz], feed_dict={
                    h.x: images_u,
                    h.z: z_true_u,
                    h.lr: learning_rate
                })

                _, loss_discriminator_img = sess.run([h.opt_dimg, h.loss_dimg], feed_dict={
                    h.x: images_u,
                    h.x_s: images_u_s,
                    h.lr: learning_rate
                })

                # adversarial phase for generator
                _, loss_generator_z = sess.run([h.opt_e, h.loss_e], feed_dict={
                    h.x: images_u,
                    h.lr: learning_rate
                })

                _, loss_generator_img = sess.run([h.opt_d, h.loss_d], feed_dict={
                    h.x: images_u,
                    h.lr: learning_rate
                })

                sum_loss_reconstruction += loss_reconstruction
                sum_loss_discrminator_z += loss_discriminator_z
                sum_loss_discrminator_img += loss_discriminator_img
                sum_loss_generator_z += loss_generator_z
                sum_loss_generator_img += loss_generator_img

                if i % 1000 == 0:
                    process_iteration.show_table_2d(i, num_trains_per_epoch, {
                        'reconstruction': sum_loss_reconstruction / (i + 1),
                        'discriminator_z': sum_loss_discrminator_z / (i + 1),
                        'discriminator_img': sum_loss_discrminator_img / (i + 1),
                        'generator_z': sum_loss_generator_z / (i + 1),
                        'generator_img': sum_loss_generator_img / (i + 1),
                    })

            average_loss_per_epoch = [
                sum_loss_reconstruction / num_trains_per_epoch,
                sum_loss_discrminator_z / num_trains_per_epoch,
                sum_loss_discrminator_img / num_trains_per_epoch,
                sum_loss_generator_z / num_trains_per_epoch,
                sum_loss_generator_img / num_trains_per_epoch,
                (sum_loss_discrminator_z+sum_loss_discrminator_img)/num_trains_per_epoch,
                (sum_loss_generator_z + sum_loss_generator_img) / num_trains_per_epoch
            ]
            training_epoch_loss.append(average_loss_per_epoch)
            training_loss_name = [
                'reconstruction',
                'discriminator_z',
                'discriminator_img',
                'generator_z',
                'generator_img',
                'discriminator',
                'generator'
            ]

            if epoch % 1 == 0:
                process.show_bar(epoch, max_epoch, {
                    'loss_r': average_loss_per_epoch[0],
                    'loss_d': average_loss_per_epoch[5],
                    'loss_g': average_loss_per_epoch[6]
                })

                plt.scatter_labeled_z(sess.run(h.z_r, feed_dict={h.x: images[:1000]}), [int(var) for var in labels[:1000]],
                                      dir=config.log_dir,
                                      filename='z_representation-{}'.format(epoch))

            if epoch % 10 == 0:
                saver.save(sess, os.path.join(config.ckpt_dir, 'model_ckptpoint'), global_step=epoch)
                pickle.dump((training_epoch_loss, training_loss_name), open(config.ckpt_dir + '/pickle.pkl', 'wb'))

if __name__ == '__main__':
    main()
