import os, sys
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dataset
from aae import Config
from model import build_graph
from plot import scatter_labeled_z, tile_images, plot_loss_tendency
config = Config()

def main():
    images, labels = dataset.load_test_images()
    num_scatter = len(images)
    _images, _, label_id = dataset.sample_labeled_data(images, labels, num_scatter)
    with tf.device(config.device):
        x, z_respresentation, x_construction = build_graph(is_test=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_dir))
        z, _x = sess.run([z_respresentation, x_construction], feed_dict={x : _images})

        scatter_labeled_z(z, label_id, dir=config.ckpt_dir)
        tile_images(_x[:100], dir=config.ckpt_dir)
        plot_loss_tendency(config.ckpt_dir)


if __name__ == '__main__':
    main()





