import os
import scipy.io as scio

from data_providers.base_provider import SpectraDataset, DataProvider

class OilDataset(SpectraDataset):
    def __init__(self, images, labels, shuffle):
        """
        :param images: 4D numpy array
        :param labels: 2D or 1D numpy array
        :param shuffle: 'str' or None:
                         - None : no any shuffling
                         - once_prior_train : shuffle data only  once prior train
                         - every_epoch : shuffle train data prior every epoch
        """
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == "once_prior_epoch":
            self.shuffle_every_epoch = False
            images, labels = self.shuffle_spectra_and_labels(images, labels)
        elif shuffle == "every_epoch":
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_spectra_and_labels(
                self.images, self.labels)
        else:
            images, labels = self.images, self.labels

        self.epoch_images = images
        self.epoch_labels = labels

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        return images_slice, labels_slice



class OilDataProvider(DataProvider):
    """Abstract class for oil readers"""

    def __init__(self, save_path=None, test_split=None, validation_set=None,
                 validation_split=None, shuffle=None,
                 **kwargs):
        """
        Args:
            save_path:  'str'
            validation_set: 'bool'
            validation_split: 'float' or None
                 - float  ; chunk of 'train set' will be marked as 'validation set'
                 - None ; if 'validation set ' == True, 'validation set will be a copy of test set
            shuffle: 'str' or None
                 - None ; no any shuffling
                 - once_prior_Train : shuffle train data only prior train
                 - every_epoch : shuffle train data every epoch
            normalization: 'str' or None
                 - None ; no any normalizaiton
                 - min_max_scalar
                 - max_mean_scalar
        """
        self._save_path = save_path
        if test_split is None:
            self.test_split = 0.20
        self.images, self.labels = self.read_oil_mat()
        images = self.normalize_images(self.images, 'min_max_scalar')
        labels = self.normalize_labels(self.labels, 'max_mean_scalar')

        # add train and test datasets
        split_idx = int(images.shape[0] * (1 - self.test_split))
        self.train = OilDataset(
            images=images[:split_idx],
            labels=labels[:split_idx],
            shuffle=shuffle)
        self.test = OilDataset(
            images=images[split_idx:],
            labels=labels[split_idx:],
            shuffle=shuffle)

        # split the train datasets as train and validation dataset
        if validation_set is not None and validation_split is not None:
            split_idx = int(self.train.num_examples * (1 - validation_split))
            self.train = OilDataset(
                images=self.train.images[:split_idx],
                labels=self.train.labels[:split_idx],
                shuffle=shuffle)
            self.validation = OilDataset(
                images=self.train.images[split_idx:],
                labels=self.train.labels[split_idx:],
                shuffle=shuffle)

        if validation_set and not validation_split:
            self.validation = self.test

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join('./data_providers/data', 'NMR_Index.mat')
        return self._save_path

    @property
    def data_shape(self):
        return (1, 700, 1)

    def read_oil_mat(self):
        spect = scio.loadmat(self.save_path)
        images, labels = spect['NMR'], spect['Index'][:, 3]
        images = images.reshape(-1, 1, 700, 1)
        labels = labels.reshape(-1, 1)
        return images, labels

if __name__=='__main__':
    dataprovider = OilDataProvider(validation_set=True, save_path='./data/NMR_Index.mat')
    import matplotlib.pyplot as plt

    im, label = dataprovider.train.next_batch(batch_size=64)
    im = im.reshape(64, -1)
    plt.plot(im.T, '-')
    plt.show()

    plt.plot(label, '-r', ms=4)
    plt.show()
    assert dataprovider.images.shape == (479, 1, 700, 1)
    assert dataprovider.train.images.shape == (383, 1, 700, 1)
    assert dataprovider.test.images.shape == (96, 1, 700, 1)
    assert dataprovider.test.labels.shape == (96, 1)
    assert dataprovider.train.labels.shape == (383, 1)