import os, sys
import tensorflow as tf
import pylab
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dataset
from model import build_graph, config
import plot
from aae_classification import Operation

opt = Operation()
images, labels = dataset.load_test_images()
num_clusters = config.ndim_y
num_plots_per_cluster = 11
image_width = 28
image_height = 28
ndim_x = config.ndim_x
pylab.gray()

with tf.device(config.device):
    t = build_graph(is_test=True)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_dir))

    # plot_cluster head
    head_y = np.identity(config.ndim_y, dtype=np.float32)
    head_z = np.zeros((config.ndim_y, config.ndim_z), dtype=np.float32)
    head_x = sess.run(t.x_r, feed_dict={t.y_r: head_y, t.z_r: head_z})
    head_x = (head_x + 1)/2
    for n in range(config.ndim_y):
        pylab.subplot(num_clusters, num_plots_per_cluster + 2, n * (num_plots_per_cluster + 2) + 1)
        pylab.imshow(head_x[n].reshape((image_width, image_height)), interpolation='none')
        pylab.axis('off')

    # plot elements in cluster
    counter_cluster = [0 for i in range(num_clusters)]
    specimen_indices = np.arange(len(images))
    np.random.shuffle(specimen_indices)
    batch_size = 500

    location_num = 0
    x_batch =  np.zeros((batch_size, ndim_x), dtype=np.float32)
    for n in range(int(len(images) / batch_size)):
        for batch in range(batch_size):
            x_batch[batch] = images[specimen_indices[location_num]]
            location_num += 1

        y_onehot = sess.run(t.y_r, feed_dict={t.x: x_batch})
        y_labels = opt.argmax_label_from_distribution(y_onehot)
        for m in range(batch_size):
            cluster = int(y_labels[m])
            counter_cluster[cluster] += 1
            if counter_cluster[cluster] < num_plots_per_cluster:
                x = (x_batch[m] + 1.0)/2
                pylab.subplot(num_clusters, num_plots_per_cluster+2, cluster * (num_plots_per_cluster+2) + counter_cluster[cluster] + 2)
                pylab.imshow(x.reshape((image_width, image_height)),interpolation='none')
                pylab.axis('off')

    fig = pylab.gcf()
    fig.set_size_inches(num_plots_per_cluster, config.ndim_y)
    pylab.savefig('{}/analogy.png'.format(config.ckpt_dir))
    print(counter_cluster)

hist_value, hist_head = plot.load_pickle_to_data(config.ckpt_dir)
for loss_name in ['reconstruction']:
    plot.plot_loss_trace(hist_value[loss_name], loss_name, config.ckpt_dir)

plot.plot_adversarial_trace(hist_value['discriminator_y'], hist_value['generator_y'], 'y', config.ckpt_dir)
plot.plot_adversarial_trace(hist_value['discriminator_z'], hist_value['generator_z'], 'z', config.ckpt_dir)







