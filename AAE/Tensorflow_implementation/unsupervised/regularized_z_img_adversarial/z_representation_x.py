import os, sys
import tensorflow as tf
import numpy as np
import pylab

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from model import build_graph, config


with tf.device(config.device):
    t = build_graph(is_test=True)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_dir))
    # z distributed plot
    num_segments = 20
    xlimit = (-6, 2)
    ylimit = (-6, 2)
    x_values = np.linspace(xlimit[0], xlimit[1], num_segments)
    y_values = np.linspace(ylimit[0], ylimit[1], num_segments)
    vacant = np.zeros((28 * num_segments, 28 * num_segments))
    z_batch = []
    for x_element in x_values:
        for y_element in y_values:
            z_batch.append([x_element, y_element])

    z_batch = np.array(z_batch).reshape(-1, 2)

    x_reconstruction = sess.run(t.x_r, feed_dict={t.z_r: z_batch})
    x_reconstruction = (x_reconstruction + 1)/2
    pylab.figure(figsize=(10, 10), dpi=400, facecolor='white')
    for i in range(len(x_reconstruction)):
        m = i // num_segments
        n = i % num_segments
        pylab.subplot(num_segments, num_segments, m * num_segments + n + 1)
        pylab.imshow(x_reconstruction[i].reshape(28, 28), cmap='gray')
        pylab.axis('off')

    pylab.savefig("{}/clusters.png".format(config.ckpt_dir))

