import numpy as np


class Dataset:
    """Classs to represent some dataset: train, validation , test"""

    @property
    def num_examples(self):
        """Return numbers of examples in dataset"""
        raise NotImplementedError

    def next_batch(self, batch_size):
        """Return batch of required size of data, labels"""
        raise NotImplementedError


class SpectraDataset(Dataset):
    """Dataset for images that provide some often used method"""

    @staticmethod
    def shuffle_spectra_and_labels(images, labels):
        rand_indexes = np.random.permutation(images.shape[0])
        shuffled_images = images[rand_indexes]
        shuffled_labels = labels[rand_indexes]
        return shuffled_images, shuffled_labels


class DataProvider:
    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def n_classes(self):
        raise NotImplementedError

    class data_statistic:
        def __init__(self, data):
            self.max = np.amax(data)
            self.min = np.amin(data)
            self.mean = (self.max + self.min) / 2
            self.range = (self.max - self.min) / 2

    @property
    def spectra_stat(self):
        if not hasattr(self, '_spectra_stat'):
            self._spectra_stat = self.data_statistic(self.images)
        return self._spectra_stat

    @property
    def labels_stat(self):
        if not hasattr(self, '_labels_stat'):
            self._labels_stat = self.data_statistic(self.labels)
        return self._labels_stat

    def normalize_images(self, images, normalization_type):
        """
        Args:
        :param images: numpy 4D array
        :param normalization_type:  'str', available choices
                                   - min_max_scalar                  scale data matrix to [0, 1] range
                                   - max_mean_scalar                 scale data matrix to [-1, 1] range
        """
        if normalization_type == 'min_max_scalar':
            images = (images - self.spectra_stat.min) / (2 * self.spectra_stat.range)
        elif normalization_type == 'max_mean_scalar':
            images = (images - self.spectra_stat.mean) / self.spectra_stat.range
        else:
            raise Exception("Unknown type of normalization")
        return images

    def normalize_labels(self, labels, normalization_type):
        """
        Args:
        :param images: numpy 4D array
        :param normalization_type:  'str', available choices
                                   - min_max_scalar                  scale data matrix to [0, 1] range
                                   - max_mean_scalar                 scale data matrix to [-1, 1] range
        """
        if normalization_type == 'min_max_scalar':
            labels = (labels - self.labels_stat.min) / (2 * self.labels_stat.range)
        elif normalization_type == 'max_mean_scalar':
            labels = (labels - self.labels_stat.mean) / self.labels_stat.range
        else:
            raise Exception("Unknown type of normalization")
        return labels
