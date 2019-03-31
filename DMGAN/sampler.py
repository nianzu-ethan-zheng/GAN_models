"""
Author: Nianzu Ethan Zheng
Datetime: 2018-2-3
Place: Shenyang China
Copyright
"""
import numpy as np
import random
from math import *


def sampler_switch(config):
    """
    Get a random sampler , which need to subject to a specific probability distribution
    Keyword arguments:
        config: Specify the type of Probability distribution and shape of output
                shorthand writing means:
                sr : swiss_roll
                gm : gaussian_mixture
                ud : uniform_desk
                g : gaussian
                u : uniform

    Return A batch whose shape is batch_size x ndim_z
    """
    if config.distribution_sampler == 'sr':
        z_true_u = swiss_roll(
            config.batch_size,
            config.ndim_z,
            n_labels=10
        )
    elif config.distribution_sampler == 'gm':
        z_true_u = gaussian_mixture(
            config.batch_size,
            config.ndim_z,
            n_labels=10
        )
    elif config.distribution_sampler == 'ud':
        z_true_u = uniform_desk(
            config.batch_size,
            config.ndim_z,
            radius=1.0,
        )
    elif config.distribution_sampler == 'gs':
        z_true_u = gaussian(
            config.batch_size,
            config.ndim_z,
            mean=0,
            var=0.4,
        )
    elif config.distribution_sampler == 'us':
        z_true_u = uniform(
            config.batch_size,
            config.ndim_z,
            minv=-1,
            maxv=1
        )
    else:
        raise Exception
    return z_true_u


def uniform(batch_size, n_dim, minv=-1.0, maxv=1.0):
    # n_dim           ---> dimension of z
    return np.random.uniform(minv, maxv, (batch_size, n_dim)).astype(np.float32)


def gaussian(batch_size, n_dim, mean=0, var=0.5):
    return np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)


def uniform_desk(batch_size, n_dim, radius=1.0):
    def get_normalized_vector(v):
        v = v / np.tile((1e-20 + np.max(np.abs(v), axis=1)).reshape(-1, 1), (1, 2))
        v_2 = np.tile(np.sum(v**2, axis=1).reshape(-1, 1), (1, 2))
        return v / np.sqrt(1e-6 + v_2)

    v = np.random.normal(loc=0, scale=1, size=(batch_size, n_dim))
    v = get_normalized_vector(v)
    r = np.random.uniform(low=0, high=1, size=(batch_size, 1)) ** (1 / n_dim)
    r = np.hstack((r, r))
    return radius * v * r


def gaussian_mixture(batch_size, n_dim, n_labels=10):
    if n_dim % 2 != 0:
        raise Exception('n_dim must be a multiple of 2')

    def sample(x, y, label, n_labels):

        # label       ---> specified label number

        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * label
        # rotation and shift formula
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batch_size, int(n_dim / 2)))
    y = np.random.normal(0, y_var, (batch_size, int(n_dim / 2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(int(n_dim / 2)):
            z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], random.randint(0, n_labels - 1), n_labels)

    return z


def supervised_gaussian_mixture(batch_size, n_dim, label_indices, n_labels):
    if n_dim % 2 != 0:
        raise Exception('n_dim must be a multiple of 2')

    def sample(x, y, label, n_labels):

        # label       ---> specified label number

        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * label
        # rotation and shift formula
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batch_size, int(n_dim / 2)))
    y = np.random.normal(0, y_var, (batch_size, int(n_dim / 2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(int(n_dim / 2)):
            z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)

    return z


def swiss_roll(batch_size, n_dim, n_labels=10):
    def sample(label, n_labels):
        uni = np.random.uniform(0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape(2, )

    z = np.zeros((batch_size, int(n_dim)), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(int(n_dim / 2)):
            z[batch, zi * 2:zi * 2 + 2] = sample(random.randint(0, n_labels - 1), n_labels)
    return z


def supervised_swiss_roll(batch_size, n_dim, label_indices, n_labels):
    def sample(label, n_labels):
        uni = np.random.uniform(0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(int(n_dim / 2)):
            z[batch, zi * 2:zi * 2 + 2] = sample(label_indices[batch], n_labels)

    return z
