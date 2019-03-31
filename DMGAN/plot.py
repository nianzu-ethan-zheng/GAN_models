"""
Author: Nianzu Ethan Zheng
Datetime: 2018-2-2
Place: Shenyang China
Copyright
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
plt.style.use('classic')


def scatter_z(z_batch, path=None, filename='z'):
    if path is None:
        raise Exception('please try a valid folder')

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(20.0, 16.0)
    for n in range(z_batch.shape[0]):
        plt.scatter(z_batch[n, 0], z_batch[n, 1], s=600, marker="o", edgecolors='none')
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.savefig("{}/{}.png".format(path, filename))


def scatter_labeled_z(z_batch, label_batch, dir=None, filename='labeled_z'):
    """
    Get scatter plot
    Keyword arguments:
        z_batch: A Nx2 numpy array
        label_batch: A Nx2 numpy array
        dir:  Path that save picture
        filename: just like literal meaning
    Return a plot
    """
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(20, 16)

    def stairing_array(array):
        levels = []
        for i in range(len(array)):
            if array[i] < np.percentile(array, 10):
                level = 0
            elif array[i] < np.percentile(array, 20):
                level = 0
            elif array[i] < np.percentile(array, 30):
                level = 1
            elif array[i] < np.percentile(array, 40):
                level = 1
            elif array[i] < np.percentile(array, 50):
                level = 2
            elif array[i] < np.percentile(array, 60):
                level = 2
            elif array[i] < np.percentile(array, 70):
                level = 3
            elif array[i] < np.percentile(array, 80):
                level = 3
            elif array[i] < np.percentile(array, 90):
                level = 4
            else:
                level = 4
            levels.append(level)
        return np.array(levels)

    colors = ["#2103c8", "#0e960e", "#e40402", "#05aaa8", "#ac02ab",
              "#aba808", "#151515", "#94a169", "#bec9cd", "#6a6551"]

    def distance(sequence):
        return [math.sqrt(sequence[n, 0] ** 2 + sequence[n, 1] ** 2) for n in range(sequence.shape[0])]

    label_batch = stairing_array(distance(label_batch))
    for n in range(z_batch.shape[0]):
        plt.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[int(label_batch[n])], s=1300, marker='.',
                      edgecolors='none')
    plt.xlabel('z1', fontsize=9)
    plt.ylabel('z2', fontsize=9)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.savefig('{}/{}.png'.format(dir, filename))


class Visualizer:
    def tsplot(self, x, name, dir):
        """Time series plot
        """
        plt.clf()
        plt.figure(figsize=(6, 4), dpi=500, facecolor='white')
        plt.plot(x, '-r*', ms=2, linewidth=1)
        plt.legend(name, fontsize=10)
        plt.ylabel(name + 'loss per epoch')
        plt.xlabel('Epoch', fontsize=9)
        plt.savefig('{}/{}.png'.format(dir, 'Loss-' + name),  dpi=500)
        plt.close()

    def dyplot(self, x, y, name, dir):
        """double y axis plot
        """
        fig, ax1 = plt.subplots(figsize=(6, 4), dpi=500, facecolor='white')
        ax1.plot(x, '-b*', ms=2, linewidth=1)
        ax1.set_xlabel('Epoch', fontsize=9)
        ax1.set_ylabel('Discriminator Loss per Epoch', fontsize=9, color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot( y, '-r*', ms=2, linewidth=1)
        ax2.set_ylabel('Generator Loss per Epoch', fontsize=9, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        plt.savefig('{}/{}.png'.format(dir, 'Loss-Adversarial-' + name))
        plt.close()

    def cplot(self, x, y, name, dir):
        """contrast diagram
        """
        plt.clf()
        plt.figure(figsize=(6, 4), dpi=500, facecolor='white')
        plt.plot(x, '-r*', ms=2, linewidth=1)
        plt.plot(y, '-b*', ms=2, linewidth=1)
        plt.legend(name[1:], fontsize=10)
        plt.ylabel(name[0] + ' per epoch')
        plt.xlabel('Time /2h', fontsize=9)
        plt.savefig('{}/{}.png'.format(dir, 'Loss-' + name[0]))
        plt.close()

    def mtsplot(self, y, name, dest, dir):
        """multiple diagrams
        """
        plt.clf()
        plt.figure(figsize=(6, 4), dpi=500, facecolor='white')
        colors = ["#e40402", "#2103c8", "#0e960e", "#ac02ab", "#aba808", "#05aaa8", "#151515", "#94a169", "#bec9cd",
                  "#6a6551"]
        for n in range(y.shape[1]):
            plt.plot(y[:, n], c=colors[n], linewidth=1, marker='*', markersize=2)
        plt.legend(name, fontsize=10)
        plt.xlabel('Time /2h', fontsize=9)
        plt.ylabel(dest + ' value', fontsize=9)
        plt.title(dest)
        plt.savefig('{}/{}.png'.format(dir, 'Comparision_' + dest))
        plt.close()

    def kdeplot(self, x, y, _dir, name='latent_space'):
        sns.set(style='dark')
        f, ax = plt.subplots(figsize=(6, 6))
        cmap = sns.cubehelix_palette(n_colors=6, start=1, light=1, rot=0.4, as_cmap=True)
        sns.kdeplot(x, y, cmap='Blues', shade=True, cut=5, ax=ax)
        f.savefig('{}/{}.png'.format(_dir, name))
        plt.close()

    def jointplot(self, x, y, _dir, kind="scatter", name="latent_space"):
        sns.set(style='white')
        g = sns.jointplot(x, y, kind=kind, size=6, space=0, color='b')
        plt.savefig('{}/{}.png'.format(_dir, name))

