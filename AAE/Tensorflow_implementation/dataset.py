import mnist_tools
import numpy as np
import os
from aae_dim_reduction import Config
import operator, functools


# auxiliary function
def nCr(n, r):
    r = min(n - r, r)
    if r == 0: return 1
    numer = functools.reduce(operator.mul, range(n, n - r, -1))
    denom = functools.reduce(operator.mul, range(1, r + 1, 1))
    return int(numer / denom)


def cluster_create_dataset(ndim_y):
    # list all possible combinations of two cluster heads
    num_combination = nCr(ndim_y, 2)
    '''
    i * (i-1) / 2           ---> staring point
    n                       ---> numerate scope
    '''
    # starting labels
    # [0, 1, 0, 0]
    # [0, 0, 1, 0]
    # [0, 0, 1, 0]
    # [0, 0, 0, 1]
    # [0, 0, 0, 1]
    # [0, 0, 0, 1]
    starting_labels = np.zeros((num_combination, ndim_y), dtype=np.float)
    for i in range(1, ndim_y):
        for n in range(i):
            j = int(i * (i - 1) / 2 + n)
            starting_labels[j, i] = 1

    # ending labels
    # [1, 0, 0, 0]
    # [1, 0, 0, 0]
    # [0, 1, 0, 0]
    # [1, 0, 0, 0]
    # [0, 1, 0, 0]
    # [0, 0, 1, 0]
    ending_labels = np.zeros((num_combination, ndim_y), dtype=np.float32)
    for i in range(1, ndim_y):
        for n in range(i):
            j = int(i * (i - 1) / 2 + n)
            ending_labels[j, n] = 1

    return starting_labels, ending_labels


minist_path = '../../mnist/'


def load_train_images():
    if not os.path.exists(minist_path + 'training_images.npy'):
        images, labels = mnist_tools.load_train_images()
        np.save(minist_path + 'training_images.npy', images)
        np.save(minist_path + 'training_labels.npy', labels)
    images = np.load(minist_path + 'training_images.npy')
    labels = np.load(minist_path + 'training_labels.npy')
    return images, labels


def load_test_images():
    if not os.path.exists(minist_path + 'testing_images.npy'):
        images, labels = mnist_tools.load_test_images()
        np.save(minist_path + 'testing_images.npy', images)
        np.save(minist_path + 'testing_labels.npy', labels)
    images = np.load(minist_path + 'testing_images.npy')
    labels = np.load(minist_path + 'testing_labels.npy')
    return images, labels


def create_semisupervised(images, labels, num_validation_data=10000, num_labeled_data=100, num_types_of_label=10):
    assert len(images) >= num_validation_data + num_labeled_data
    '''
    indices_for_label               ---> a dictionary contain keys, each keys corresponds to many selected images
    '''

    train_labeled_x = []
    train_unlabeled_x = []
    validation_x = []
    validation_labels = []
    train_labels = []
    indices_for_label = {}
    num_data_per_label = int(num_labeled_data / num_types_of_label)
    num_unlabeled_data = len(images) - num_validation_data - num_labeled_data

    indices = np.arange(len(images))
    np.random.shuffle(indices)

    def check(index):
        label = labels[index]
        if label not in indices_for_label:
            indices_for_label[label] = []
            return True

        if len(indices_for_label[label]) < num_data_per_label:
            for i in indices_for_label[label]:
                if i == index:
                    return False
            return True
        return False

    # label data  --->  unlabeled data --->  validation data

    for n in range(len(images)):
        index = indices[n]
        if check(index):
            indices_for_label[labels[index]].append(index)
            train_labeled_x.append(images[index])
            train_labels.append(labels[index])
        else:
            if len(train_unlabeled_x) < num_unlabeled_data:
                train_unlabeled_x.append(images[index])
            else:
                validation_x.append(images[index])
                validation_labels.append(labels[index])

    # Convert to numpy array
    train_labeled_x = np.asanyarray(train_labeled_x)
    train_unlabeled_x = np.asanyarray(train_unlabeled_x)
    validation_x = np.asanyarray(validation_x)
    validation_labels = np.asanyarray(validation_labels)
    train_labels = np.asanyarray(train_labels)

    return train_labeled_x, train_labels, train_unlabeled_x, validation_x, validation_labels


def sample_labeled_data(images, label_dights, batch_size, ndim_x=28 ** 2, ndim_y=10):
    image_batch = np.zeros((batch_size, ndim_x), dtype=np.float32)
    label_onehot_batch = np.zeros((batch_size, ndim_y), dtype=np.float32)
    label_id_batch = np.zeros((batch_size,), dtype=np.float32)
    indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batch_size, replace=False)
    for j in range(batch_size):
        data_index = indices[j]
        image_batch[j] = images[data_index]
        label_onehot_batch[j, int(label_dights[data_index])] = 1
        label_id_batch[j] = int(label_dights[data_index])
    return image_batch, label_onehot_batch, label_id_batch


def sample_unlabeled_data(images, batch_size, ndim_x=28 ** 2):
    image_batch = np.zeros((batch_size, ndim_x), dtype=np.float32)
    indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batch_size, replace=False)
    for j in range(batch_size):
        data_index = indices[j]
        image_batch[j] = images[data_index]
    return image_batch
