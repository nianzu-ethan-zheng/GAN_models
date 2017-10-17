import os, sys
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dataset
from aae import Config
from model import build_graph
import plot
config = Config()

def main():
    images, labels = dataset.load_test_images()
    num_scatter = len(images)
    _images, _, label_id = dataset.sample_labeled_data(images, labels, num_scatter)
    with tf.device(config.device):
        t = build_graph(is_test=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_dir))
        z, _x = sess.run([t.z_r, t.x_r], feed_dict={t.x: _images})

        plot.scatter_labeled_z(z, label_id, dir=config.ckpt_dir)
        plot.tile_images(_x[:100], dir=config.ckpt_dir)
        plot.plot_loss_tendency(config.ckpt_dir)

    hist_value, hist_head = plot.load_pickle_to_data(config.ckpt_dir)
    for loss_name in ['reconstruction']:
        plot.plot_loss_trace(hist_value[loss_name], loss_name, config.ckpt_dir)

    plot.plot_adversarial_trace(hist_value['discriminator'], hist_value['generator'], 'z', config.ckpt_dir)
    plot.plot_adversarial_trace(hist_value['discriminator_z'], hist_value['generator_z'], 'z', config.ckpt_dir)
    plot.plot_adversarial_trace(hist_value['discriminator_img'], hist_value['generator_img'], 'z', config.ckpt_dir)


if __name__ == '__main__':
    main()





