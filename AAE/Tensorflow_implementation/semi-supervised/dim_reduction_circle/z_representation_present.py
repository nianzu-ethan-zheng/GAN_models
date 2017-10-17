import os, sys
import tensorflow as tf
import numpy as np
import pylab

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dataset
from model import build_graph, config
import plot

def main():
    # load MNIST images
    images, labels = dataset.load_test_images()

    # Settings
    num_scatter = len(images)
    _images, _, label_id = dataset.sample_labeled_data(images, labels, num_scatter)

    with tf.device(config.device):
        t = build_graph(is_test=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_dir))

        num_segments = 40
        limit = (2, 3)
        x_values = np.linspace(limit[0], limit[1], num_segments)
        y_values = np.linspace(limit[0], limit[1], num_segments)
        vacant = np.zeros((28 * num_segments, 28 * num_segments))
        for i, x_element in enumerate(x_values):
            for j, y_element in enumerate(y_values):
                x_reconstruction = sess.run(t.x_r, feed_dict={t.yz: np.reshape([x_element, y_element], [1, 2])})
                vacant[(num_segments - 1 - i) * 28:(num_segments - i) * 28,
                j * 28:(j + 1) * 28] = x_reconstruction.reshape(28, 28)

        vacant = (vacant + 1) / 2
        pylab.figure(figsize=(10, 10), dpi=400, facecolor='white')
        pylab.imshow(vacant, cmap='gray', origin='upper')
        pylab.tight_layout()
        pylab.axis('off')
        pylab.savefig("{}/clusters.png".format(config.ckpt_dir))


if __name__ == '__main__':
    main()





