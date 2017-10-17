import urllib.request as request
import os, gzip
import numpy as np

parent = "http://yann.lecun.com/exdb/mnist"
train_images_filename = "train-images-idx3-ubyte.gz"
train_labels_filename = "train-labels-idx1-ubyte.gz"
test_images_filename = "t10k-images-idx3-ubyte.gz"
test_labels_filename = "t10k-labels-idx1-ubyte.gz"
n_train = 60000
n_test = 10000
dim = 28 * 28

_dir = '../../mnist'
if not os.path.exists(_dir):
    os.mkdir(_dir)


def load_train_images():
    if not os.path.exists(_dir + '/' + train_images_filename):
        download_data()
    images, labels = load_mnist(_dir + '/' + train_images_filename, _dir + '/' + train_labels_filename, n_train)
    return images, labels


def load_test_images():
    if not os.path.exists(_dir + '/' + test_images_filename):
        download_data()
    images, labels = load_mnist(_dir + '/' + test_images_filename, _dir + '/' + test_labels_filename, n_test)
    return images, labels


def load_mnist(data_file_path, label_file_path, num):
    images = np.zeros((num, dim), dtype=np.float32)
    labels = np.zeros((num,), dtype=np.float32)
    with gzip.open(data_file_path, 'rb') as f_images, gzip.open(label_file_path, 'rb') as f_labels:
        f_images.read(16)
        f_labels.read(8)
        for i in range(num):
            labels[i] = ord(f_labels.read(1))
            for j in range(dim):
                images[i, j] = ord(f_images.read(1))

    images = (images / 255 * 2) - 1
    return images, labels


def download_data():
    def retrieve(parent, filename):
        print('Downloading {}...'.format(filename))
        request.urlretrieve('{}/{}'.format(parent, filename), '{}/{}'.format(_dir, filename))

    for filename in [train_images_filename, train_labels_filename, test_images_filename, test_labels_filename]:
        if not os.path.exists(_dir + '/' + filename):
            retrieve(parent, filename)
