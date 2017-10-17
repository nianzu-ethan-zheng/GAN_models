import pylab
import matplotlib.patches as mpatches
import math
import numpy as np
import _pickle as pickle
import pandas as pd


def tile_images(image_batch, image_width=28, image_height=28, dir=None, filename='images'):
    if dir is None:
        raise Exception('please provide a path ')

    pylab.clf()
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.gray()

    width = int(math.sqrt(len(image_batch)))
    for n in range(width ** 2):
        pylab.subplot(width, width, n + 1)
        pylab.imshow(image_batch[n].reshape((image_width, image_height)))
        pylab.axis('off')
    pylab.savefig('{}/{}.png'.format(dir, filename))


def scatter_z(z_batch, dir=None, filename='z'):
    if dir == None:
        raise Exception('No dir exits')

    pylab.clf()
    fig = pylab.gcf()
    fig.set_size_inches(20.0, 16.0)
    for n in range(z_batch.shape[0]):
        pylab.scatter(z_batch[n, 0], z_batch[n, 1], s=40, marker="o", edgecolors='none')
    pylab.xlabel("z1")
    pylab.ylabel("z2")
    pylab.savefig("{}/{}.png".format(dir, filename))


def scatter_labeled_z(z_batch, label_batch, dir=None, filename='labeled_z'):
    pylab.clf()
    fig = pylab.gcf()
    fig.set_size_inches(20, 16)
    colors = ["#2103c8", "#0e960e", "#e40402", "#05aaa8", "#ac02ab", "#aba808", "#151515", "#94a169", "#bec9cd",
              "#6a6551"]
    for n in range(z_batch.shape[0]):
        pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[int(label_batch[n])], s=40, marker='o',
                      edgecolors='none')

    def count(label_batch):
        unique_list = []
        for label in label_batch:
            if label not in unique_list:
                unique_list.append(label)
        return len(unique_list)

    classes = [str(var) for var in np.arange(count(label_batch))]
    recs = []
    for i in range(count(label_batch)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i]))

    ax = pylab.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(recs, classes, loc='center left', bbox_to_anchor=(1.1, 0.5))
    #pylab.xticks(pylab.arange(-4, 5))
    #pylab.yticks(pylab.arange(-4, 5))
    pylab.xlabel('z1')
    pylab.ylabel('z2')

    pylab.savefig('{}/{}.png'.format(dir, filename))


def load_pickle_to_data(save_dir):
    hist_value, loss_head = pickle.load(open(save_dir + '/pickle.pkl', 'rb'))
    hist_pd = pd.DataFrame(hist_value, columns=loss_head)
    return hist_pd, loss_head


def plot_loss_tendency(save_dir):
    hist_pd, _ = load_pickle_to_data(save_dir)
    pylab.clf()
    plot_loss_trace(hist_pd['reconstruction'], loss_name='Reconstruction', save_dir=save_dir)
    pylab.close()


def plot_double_scale_trend(save_dir):
    hist_pd, _ = load_pickle_to_data(save_dir)
    plot_adversarial_trace(hist_pd['discriminator'], hist_pd['generator'], 'Total', save_dir)
    pylab.close()


def plot_loss_trace(loss_data, loss_name, save_dir):
    pylab.clf()
    pylab.figure(figsize=(6, 4), dpi=500, facecolor='white')
    pylab.plot(loss_data, '-r*', ms=2, linewidth=1)
    pylab.legend(loss_name, fontsize=10)
    pylab.ylabel(loss_name + 'loss per epoch')
    pylab.xlabel('Epoch', fontsize=9)
    pylab.savefig('{}/{}.png'.format(save_dir, 'Loss-' + loss_name))


def plot_adversarial_trace(loss_d, loss_g, loss_name, save_dir):
    fig, ax1 = pylab.subplots(figsize=(6, 4), dpi=500, facecolor='white')
    ax1.plot(loss_d, '-b*', ms=2, linewidth=1)
    ax1.set_xlabel('Epoch', fontsize=9)
    ax1.set_ylabel('Discriminator Loss per Epoch', fontsize=9, color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(loss_g, '-r*', ms=2, linewidth=1)
    ax2.set_ylabel('Generator Loss per Epoch', fontsize=9, color='r')
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    pylab.savefig('{}/{}.png'.format(save_dir, 'Loss-Adversarial-' + loss_name))

