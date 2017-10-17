import os, sys
import tensorflow as tf
import pylab
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dataset
from model import build_graph, config
import plot

def main():
    # load MNIST images
    images, labels = dataset.load_test_images()

    # Settings
    num_anologies = 10
    pylab.gray()

    # generate style vector z
    x = dataset.sample_unlabeled_data(images, num_anologies)
    x = (x + 1)/2

    with tf.device(config.device):
        x_input, img_y, img_z, reconstruction = build_graph(is_test=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_dir))
        z = sess.run(img_z, feed_dict={x_input: x})

        for m in range(num_anologies):
            pylab.subplot(num_anologies, config.ndim_y+2, m*12+1)
            pylab.imshow(x[m].reshape((28, 28)), interpolation='none')
            pylab.axis('off')

        all_y = np.identity(config.ndim_y, dtype=np.float32)
        for m in range(num_anologies):
            fixed_z = np.repeat(z[m].reshape(1, -1), config.ndim_y, axis=0)
            gen_x = sess.run(reconstruction, feed_dict={img_z: fixed_z, img_y: all_y})
            gen_x = (gen_x + 1)/2

            for n in range(config.ndim_y):
                pylab.subplot(num_anologies, config.ndim_y+2, m*12 + 3 +n)
                pylab.imshow(gen_x[n].reshape((28, 28)),interpolation='none')
                pylab.axis('off')

        fig = pylab.gcf()
        fig.set_size_inches(num_anologies, config.ndim_y)
        pylab.savefig('{}/analogy.png'.format(config.ckpt_dir))

        hist_value, hist_head = plot.load_pickle_to_data(config.ckpt_dir)
        for loss_name in ['reconstruction', 'validation_accuracy', 'supervised']:
            plot.plot_loss_trace(hist_value[loss_name], loss_name, config.ckpt_dir)

        plot.plot_adversarial_trace(hist_value['discriminator_y'], hist_value['generator_y'], 'y', config.ckpt_dir)
        plot.plot_adversarial_trace(hist_value['discriminator_z'], hist_value['generator_z'], 'z', config.ckpt_dir)


if __name__ == '__main__':
    main()





