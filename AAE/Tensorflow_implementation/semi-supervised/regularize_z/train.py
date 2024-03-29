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
    # config
    opt = Operation()
    opt.check_dir(config.ckpt_dir, is_restart=False)
    opt.check_dir(config.log_dir, is_restart=True)

    max_epoch = 510
    num_trains_per_epoch = 500
    batch_size_l = 100
    batch_size_u = 100

    # create semi-supervised split
    # Load minist images
    images, labels = dataset.load_train_images()
    num_labeled_data = 10000
    num_types_of_label = 11  # additional label corresponds to unlabeled data
    training_images_l, training_labels_l, training_images_u, _, _ = dataset.create_semisupervised(images, labels, 0, num_labeled_data, num_types_of_label)

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
            learning_rate = opt.ladder_learning_rate(epoch + len(training_epoch_loss))

            # Recording loss per iteration
            sum_loss_reconstruction = 0
            sum_loss_discrminator = 0
            sum_loss_generator = 0
            process_iteration = Process()
            for i in range(num_trains_per_epoch):
                process_iteration.start_epoch(i, num_trains_per_epoch)
                # Inputs
                '''
                _l -> labeled
                _u -> unlabeled

                '''
                images_l, label_onehot_l, label_id_l = dataset.sample_labeled_data(training_images_l, training_labels_l, batch_size_l, ndim_y=num_types_of_label)
                images_u = dataset.sample_unlabeled_data(training_images_u, batch_size_u)
                onehot = np.zeros((1, num_types_of_label), dtype=np.float32)
                onehot[-1] = 1
                label_onehot_u = np.repeat(onehot, batch_size_u, axis=0)
                z_true_l = sampler.supervised_swiss_roll(batch_size_l, config.ndim_z, label_id_l, num_types_of_label - 1)
                z_true_u = sampler.swiss_roll(batch_size_u, config.ndim_z, num_types_of_label - 1)
                # z_true_l = sampler.supervised_gaussian_mixture(batch_size_l, config.ndim_z, label_id_l, num_types_of_label - 1)
                # z_true_u = sampler.gaussian_mixture(batch_size_u, config.ndim_z, num_types_of_label - 1)

                # reconstruction_phase
                _, loss_reconstruction = sess.run([h.opt_r, h.loss_r], feed_dict={
                    h.x: images_u,
                    h.lr: learning_rate
                })

                # adversarial phase for discriminator
                _, loss_discriminator_l = sess.run([h.opt_d, h.loss_d], feed_dict={
                    h.x: images_l,
                    h.label: label_onehot_l,
                    h.z: z_true_l,
                    h.lr: learning_rate
                })

                _, loss_discriminator_u = sess.run([h.opt_d, h.loss_d], feed_dict={
                    h.x: images_u,
                    h.label: label_onehot_u,
                    h.z: z_true_u,
                    h.lr: learning_rate
                })

                loss_discriminator = loss_discriminator_l + loss_discriminator_u

                # adversarial phase for generator
                _, loss_generator_l= sess.run([h.opt_e, h.loss_e,], feed_dict={
                    h.x: images_l,
                    h.label: label_onehot_l,
                    h.lr: learning_rate
                })

                _, loss_generator_u = sess.run([h.opt_e, h.loss_e], feed_dict={
                    h.x: images_u,
                    h.label: label_onehot_u,
                    h.lr: learning_rate
                })
                loss_generator = loss_generator_l + loss_generator_u

                sum_loss_reconstruction += loss_reconstruction / batch_size_u
                sum_loss_discrminator += loss_discriminator
                sum_loss_generator += loss_generator

                if i % 1000 == 0:
                    process_iteration.show_table_2d(i, num_trains_per_epoch, {
                        'reconstruction': sum_loss_reconstruction / (i + 1),
                        'discriminator': sum_loss_discrminator / (i + 1),
                        'generator': sum_loss_generator / (i + 1),
                    })

            average_loss_per_epoch = [
                sum_loss_reconstruction / num_trains_per_epoch,
                sum_loss_discrminator / num_trains_per_epoch,
                sum_loss_generator / num_trains_per_epoch,
            ]
            training_epoch_loss.append(average_loss_per_epoch)
            training_loss_name = [
                'reconstruction',
                'discriminator',
                'generator'
            ]

            if epoch % 1 == 0:
                process.show_bar(epoch, max_epoch, {
                    'loss_r': average_loss_per_epoch[0],
                    'loss_d': average_loss_per_epoch[1],
                    'loss_g': average_loss_per_epoch[2]
                })

                plt.tile_images(sess.run(h.x_, feed_dict={h.x: images_u}),
                                dir=config.log_dir,
                                filename='x_rec_epoch_{}'.format(str(epoch).zfill(3)))

                plt.scatter_labeled_z(sess.run(h.z_r, feed_dict={h.x: images[:1000]}), [int(var) for var in labels[:1000]],
                                      dir=config.log_dir,
                                      filename='z_representation-{}'.format(epoch))

            if epoch % 10 == 0:
                saver.save(sess, os.path.join(config.ckpt_dir, 'model_ckptpoint'), global_step=epoch)
                pickle.dump((training_epoch_loss, training_loss_name), open(config.ckpt_dir + '/pickle.pkl', 'wb'))
                plt.plot_double_scale_trend(config.ckpt_dir)


if __name__ == '__main__':
    main()
