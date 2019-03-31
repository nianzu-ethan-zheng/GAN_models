"""
Author : Nianzu Ethan Zheng
Datetime: 2018-23-9:37
Copyright
"""
import numpy as np
class Object:
    pass

class Baseset:
    """
    Provide some preliminaries and Define some requests
    """

    @property
    def num_sample(self):
        raise NotImplementedError

    def next_batch(self, batch_size):
        raise NotImplementedError

    def start_new_epoch(self):
        raise NotImplementedError

    @staticmethod
    def shuffle(subdata):
        _subdata = Object()
        rand_indexes = np.random.permutation(subdata.x.shape[0])
        for key in subdata.__dict__.keys():
            value = getattr(subdata, key)
            value = value[rand_indexes]
            _subdata.__setattr__(key, value)
        return _subdata


class Baseprovider:
    """Statistic method and Processing method"""
    
    @property
    def data_shape(self):
        raise NotImplementedError
    
    @property
    def num_sample(self):
        raise NotImplementedError

    class data_statistic:
        def __init__(self, matrix):
            self.max = np.amax(matrix)
            self.min = np.amin(matrix)
            self.mean = (self.max + self.min) / 2
            self.range = (self.max - self.min) / 2

    def normalize(self, matrix, mode):
        """
        Args:
        :param input: numpy 4D array
        :param mode:  'str', available choices
                                   - min_max_scalar                  scale data matrix to [0, 1] range
                                   - max_mean_scalar                 scale data matrix to [-1, 1] range
        """
        stats = self.data_statistic(matrix)
        if mode == 'min_max_scalar':
            matrix = (matrix - stats.min) / (2 * stats.range)
        elif mode == 'max_mean_scalar':
            matrix = (matrix - stats.mean) / stats.range
        else:
            raise Exception("Unknown type of normalization")
        return matrix

    def shape_transform(self, matrix):
        return np.array(matrix).reshape(-1, 1, 15, 1)


if __name__ == '__main__':
    b = Baseset()
    subset = Object()
    subset.x = np.arange(1, 10)
    subset.y = np.arange(1, 10)
    c = b.shuffle(subset)
    assert any(c.x == c.y)
    print(c.x, c.y)