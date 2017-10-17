# Standard library
import tensorflow as tf
import numpy as np
import _pickle as pickle
import os, sys
# Packages
from model import build_graph,config

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from aae_dim_reduction import Operation
import dataset
from process import Process
import sampler
import plot as plt


def main(run_load_from_file=False):
    # load MNIST images
    images, labels = dataset.load_train_images()

    # config
    opt = Operation()
    opt.check_dir(config.ckpt_dir, is_restart=False)
    opt.check_dir(config.log_dir, is_restart=True)

    # setting
    max_epoch = 510
    num_trains_per_epoch = 500
    batch_size_l = 100
    batch_size_u = 100

    # create semi-supervised split
    num_validation_data = 10000
    num_labeled_data = 100
    num_types_of_label = 10
    training_images_l, training_labels_l, training_images_u, validation_images, validation_labels = dataset.create_semisupervised(
        images, labels, num_validation_data, num_labeled_data, num_types_of_label)

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
            training_loss_set = []
            sum_loss_reconstruction = 0
            sum_loss_supervised = 0
            sum_loss_discrminator = 0
            sum_loss_generator = 0
            sum_loss_cluster_head = 0

            process_iteration = Process()
            for i in range(num_trains_per_epoch):
                process_iteration.start_epoch(i, num_trains_per_epoch)

                # sample from data distribution
                images_l, label_onehot_l, label_id_l = dataset.sample_labeled_data(training_images_l, training_labels_l,
                                                                                   batch_size_l)
                images_u = dataset.sample_unlabeled_data(training_images_u, batch_size_u)

                # additional cost function that penalizes the euclidean between of every two of cluster
                if epoch == 0:
                    for j in range(5):
                        starting_labels, ending_labels = dataset.cluster_create_dataset(config.ndim_y)
                        _, loss_transform = sess.run([h.opt_t, h.loss_t], feed_dict={
                            h.g_s: starting_labels,
                            h.g_e: ending_labels,
                            h.lr: learning_rate
                        })

                # reconstruction_phase
                _, loss_reconstruction = sess.run([h.opt_r, h.loss_r], feed_dict={
                    h.x: images_u,
                    h.lr: learning_rate
                })

                z_true_u = sampler.gaussian(batch_size_u, config.ndim_z, mean=0, var=1)
                y_true_u = sampler.onehot_categorical(batch_size_u, config.ndim_y)
                # adversarial phase for discriminator
                _, loss_discriminator_y = sess.run([h.opt_dy, h.loss_dy], feed_dict={
                    h.x: images_u,
                    h.y: y_true_u,
                    h.lr: learning_rate
                })

                _, loss_discriminator_z = sess.run([h.opt_dz, h.loss_dz], feed_dict={
                    h.x: images_u,
                    h.z: z_true_u,
                    h.lr: learning_rate
                })

                loss_discriminator = loss_discriminator_y + loss_discriminator_z

                # adversarial phase for generator
                _, loss_generator_y, loss_generator_z = sess.run([h.opt_e, h.loss_gy, h.loss_gz], feed_dict={
                    h.x: images_u,
                    h.lr: learning_rate
                })

                loss_generator = loss_generator_y + loss_generator_z

                # supervised phase
                _, loss_generator_supervised = sess.run([h.opt_ey, h.loss_ey], feed_dict={
                    h.x: images_l,
                    h.y_s: label_id_l,
                    h.lr: learning_rate
                })

                training_loss_set.append([
                    loss_reconstruction,
                    loss_discriminator,
                    loss_discriminator_y,
                    loss_discriminator_z,
                    loss_generator,
                    loss_generator_z,
                    loss_generator_y,
                    loss_generator_supervised,
                    loss_transform
                ])

                sum_loss_reconstruction += loss_reconstruction
                sum_loss_discrminator += loss_discriminator
                sum_loss_generator += loss_generator
                sum_loss_supervised += loss_generator_supervised
                sum_loss_cluster_head += loss_transform

                if i % 1000 == 0:
                    process_iteration.show_table_2d(i, num_trains_per_epoch, {
                        'reconstruction': sum_loss_reconstruction / (i + 1),
                        'discriminator': sum_loss_discrminator / (i + 1),
                        'generator': sum_loss_generator / (i + 1),
                        'supervise': sum_loss_supervised / (i + 1),
                        'cluster_head':sum_loss_cluster_head / (i + 1)
                    })
            # In end of epoch, summary the loss
            average_training_loss_per_epoch = np.mean(np.array(training_loss_set), axis=0)

            # validation phase
            images_v_segments = np.split(validation_images, num_validation_data // 1000)
            labels_v_segments = np.split(validation_labels, num_validation_data // 1000)
            sum_accuracy = 0
            for images_v, labels_v in zip(images_v_segments, labels_v_segments):
                y_v = sess.run(h.y_r, feed_dict={h.x: images_v})
                accuracy = opt.compute_accuracy(y_v, labels_v)
                sum_accuracy += accuracy
            validation_accuracy = sum_accuracy / len(images_v_segments)

            # append validation accuracy to the training loss
            average_loss_per_epoch = np.append(average_training_loss_per_epoch, validation_accuracy)
            training_epoch_loss.append(average_loss_per_epoch)
            loss_name_per_epoch = [
                'reconstruction',
                'discriminator',
                'discriminator_y',
                'discriminator_z',
                'generator',
                'generator_z',
                'generator_y',
                'supervised',
                'transform',
                'validation_accuracy'
            ]

            if epoch % 1 == 0:
                process.show_bar(epoch, max_epoch, {
                    'loss_r': average_loss_per_epoch[0],
                    'loss_d': average_loss_per_epoch[1],
                    'loss_g': average_loss_per_epoch[4],
                    'loss_v': average_loss_per_epoch[9],
                })

                plt.tile_images(sess.run(h.x_, feed_dict={h.x: images_u}),
                                dir=config.log_dir,
                                filename='x_rec_epoch_{}'.format(str(epoch).zfill(3)))

            if epoch % 10 == 0:
                saver.save(sess, os.path.join(config.ckpt_dir, 'model_ckptpoint'), global_step=epoch)
                pickle.dump((training_epoch_loss, loss_name_per_epoch), open(config.ckpt_dir + '/pickle.pkl', 'wb'))
                plt.plot_double_scale_trend(config.ckpt_dir)


if __name__ == '__main__':
    main()
