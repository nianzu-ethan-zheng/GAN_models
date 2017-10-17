import aae
import numpy as np

class Config(aae.Config):
    def __init__(self):
        super(Config, self).__init__()
        self.ndim_x = 28 ** 2
        self.ndim_y = 16
        self.ndim_z = 10
        self.momentum = 0.1  # setting parameters


class Operation(aae.Operation):
    def argmax_label_from_distribution(self, y_distribution):
        return np.argmax(y_distribution, axis=1)

