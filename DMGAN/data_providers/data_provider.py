import os
import scipy.io as scio
import numpy as np
from data_providers.base_provider import Baseset, Baseprovider


class Object:
    pass


class Dataset(Baseset):
    """
    Provide a sampler: given a dataset and batch_size, it output a batch of samples
    """

    def __init__(self, subset, shuffle='every_epoch'):
        """
        images: 4D numpy array
        labels: 2D or 1D numpy array
        shuffle: 'str' or None:
                 - None : no any shuffling
                 - once_prior_train : shuffle data only  once prior train
                 - every_epoch : shuffle train data prior every epoch
        """
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == "once_prior_epoch":
            self.shuffle_every_epoch = False
            subset = self.shuffle(subset)
        elif shuffle == "every_epoch":
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.subset = subset
        self.epoch_subset = []
        self.start_new_epoch()

    @property
    def num_sample(self):
        return self.subset.x.shape[0]

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            subset = self.shuffle(self.subset)
        else:
            subset = self.subset

        self.epoch_subset = subset

    def next_batch(self, batch_size):
        batch = Object()
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        value = []
        for key in self.epoch_subset.__dict__.keys():
            value = getattr(self.epoch_subset, key)[start: end]
            setattr(batch, key, value)

        if value.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        return batch


class DataProvider(Baseprovider):
    def __init__(self, save_path=None):
        self._save_path = save_path

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join('../data_providers/data', 'dataset.mat')
        return self._save_path

    def load_data(self):
        print('Loading the mat from the file path:{}'.format(self.save_path))
        data = scio.loadmat(self._save_path)
        del data['__header__']
        del data['__globals__']
        del data['__version__']
        out = Object()
        out.pi = self.shape_transform(data['P_input'])
        out.po = np.array(data['P_output'])
        out.eg = self.shape_transform(data['EG'])
        out.gc = self.shape_transform(data['G_cond'])
        out.qe = np.array(data['Q_ept'])
        out.eo = np.array(data['E_output'])

        out.keys = data.keys()
        out.dm = np.array(data['D_mid'])
        out.dr = np.array(data['D_rag'])
        out.rm = np.array(data['R_mid'])
        out.rr = np.array(data['R_rag'])
        out.qm = np.array(data['Q_mid'])
        out.qr = np.array(data['Q_rag'])
        out.rdm = np.concatenate([out.rm, out.dm], axis=1)
        out.rdr = np.concatenate([out.rr, out.dr], axis=1)
        return out


class SemiDataProvider(DataProvider):
    """Abstract class for oil readers"""

    def __init__(self, save_path=None, num_validation=74, num_label=100, shuffle='every_epoch',
                 **kwargs):
        """
        huffle: 'str' or None
        - None ; no any shuffling
        - once_prior_Train : shuffle train data only prior train
        - every_epoch : shuffle train data every epoch
        """
        super().__init__(save_path=save_path)
        self.out = self.load_data()
        data = self.create_semisupervised(self.out, num_label=num_validation, num_validation=num_label)

        # train_l, train_u, train_test
        self.train_l = Dataset(data.t_l, shuffle=shuffle)
        self.train_u = Dataset(data.t_u, shuffle=shuffle)
        self.test = data.v

    def create_semisupervised(self, data, num_validation=74, num_label=100):
        assert len(data.eg) >= num_validation + num_label
        '''
        indices_for_label  ---> a dictionary contain keys, each keys corresponds to many selected images
        '''
        indices = np.arange(len(data.eg))
        np.random.shuffle(indices)
        indices_v = indices[:num_validation]
        ul_indices = indices[num_validation:]
        indices_l = ul_indices[::5]
        indices_u = np.setdiff1d(ul_indices, indices_l, assume_unique=True)

        print("Split the data into labeled ,unlabeled, validation according to different usage")
        out = Object()
        out.t_l = Object()
        out.t_l.x = data.eg[indices_l]
        out.t_l.l = data.eo[indices_l]
        out.t_l.c = data.gc[indices_l]
        out.t_l.e = data.qe[indices_l]
        out.t_l.q = data.po[indices_l]
        out.t_l.rd = data.pi[indices_l]

        out.t_u = Object()
        out.t_u.x = data.eg[indices_u]
        out.t_u.l = data.eo[indices_u]
        out.t_u.c = data.gc[indices_u]
        out.t_u.e = data.qe[indices_u]
        out.t_u.q = data.po[indices_u]
        out.t_u.rd = data.pi[indices_u]

        out.v = Object()
        out.v.x = data.eg[indices_v]
        out.v.l = data.eo[indices_v]
        out.v.c = data.gc[indices_v]
        out.v.e = data.qe[indices_v]
        out.v.q = data.po[indices_v]
        out.v.rd = data.pi[indices_v]
        return out


class SuvsDataProvider(DataProvider):
    """Abstract class for oil readers"""

    def __init__(self, save_path=None, num_validation=74, shuffle='every_epoch',
                 **kwargs):
        """
        shuffle: 'str' or None
        - None ; no any shuffling
        - once_prior_Train : shuffle train data only prior train
        - every_epoch : shuffle train data every epoch
        """
        super().__init__(save_path=save_path)
        self.out = self.load_data()
        data = self.create_supervised(self.out, num_validation=num_validation)

        # train_l, train_u, train_test
        self.train_l = Dataset(data.t_l, shuffle=shuffle)
        self.valid = Dataset(data.v, shuffle=shuffle)
        self.test = data.v
        self.train = data.t_l

    def create_supervised(self, data, num_validation=74):
        assert len(data.eg) >= num_validation
        '''
        indices_for_label  ---> a dictionary contain keys, each keys corresponds to many selected images
        '''
        np.random.seed(2018)
        num_samples = len(data.eg)
        num_label = num_samples - num_validation
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        indices_l = indices[:num_label]
        indices_v = np.setdiff1d(indices, indices_l, assume_unique=True)

        print("Split the data into labeled ,unlabeled, validation according to different usage")
        out = Object()
        out.t_l = Object()
        out.t_l.x = data.eg[indices_l]
        out.t_l.l = data.eo[indices_l]
        out.t_l.c = data.gc[indices_l]
        out.t_l.e = data.qe[indices_l]
        out.t_l.q = data.po[indices_l]
        out.t_l.rd = data.pi[indices_l]

        out.v = Object()
        out.v.x = data.eg[indices_v]
        out.v.l = data.eo[indices_v]
        out.v.c = data.gc[indices_v]
        out.v.e = data.qe[indices_v]
        out.v.q = data.po[indices_v]
        out.v.rd = data.pi[indices_v]
        return out

if __name__ == '__main__':

    # out = data.load_data()
    # data = data.create_semisupervised(out, num_label=74, num_validation=100)
    # print(data.t_l.__dict__.keys())
    # data_label = Dataset(data.t_l, shuffle='every_epoch')
    # data_label.next_batch(64)
    # data_unlabel = Dataset(data.t_u, shuffle='every_epoch')
    # data.data_valid = data.v

    # Whether the fucntion SemiDataProvider is realized
    # data = SemiDataProvider()
    # print(type(data.out.pi))
    # print(len(data.out.pi))
    # print(data.out.pi.shape)
    # print(data.out.keys)
    # print(data.train_l.next_batch(32))
    # print(data.train_u.num_sample)

    import numpy as np
    data = SuvsDataProvider()
    # print(data.train_l.next_batch(42))
    # print(data.train_l.num_sample)
    val = np.reshape(data.test.x, [74, 15])[:, :7]
    Eng = np.sum(np.sum(np.square(val), axis=1))
    print(Eng)


    # Whether the fucntion Next_epoch is realized
    # subset = Object()
    # subset.x = np.arange(10)
    # subset.y = np.arange(10)
    # c = Dataset(subset,shuffle='every_epoch')
    # d = c.next_batch(5)
    # e = c.next_batch(5)
    # print(d.x, d.y)
    # print(e.x, e.y)
