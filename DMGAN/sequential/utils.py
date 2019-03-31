"""
Discription: A Python Progress Meter
Author: Nianzu Ethan Zheng
Date: 2018-1-21
Copyright
"""
import os
import logging
import pickle
import shutil
import pandas as pd
import tensorflow as tf
import numpy as np


def check_dir(path, is_restart=False):
    name = os.path.split(path)[1]
    if not os.path.exists(path):
        os.makedirs(path)
        print('Create a new folder named {}'.format(name))
    elif is_restart:
        shutil.rmtree(path)
        os.makedirs(path)
        print('The folder named {} is restarted'.format(name))
    print('The folder named {} has existed.'.format(name))


def copy_file(src, dst):
    shutil.copy(src, dst)
    filepath, filename = os.path.split(src)
    print("copying the {} to historical folder ".format(filename))


def pickle_load(path_name, use_pd=False):
    with open(path_name, 'rb') as f:
        value, name = pickle.load(f)
    if use_pd:
        value = pd.DataFrame(value, columns=name)
    print("pickle named {} has been loaded...".format(name))
    return value, name


def pickle_save(value, name, path_name):
    with open(path_name, 'wb') as f:
        pickle.dump((value, name), f, protocol=pickle.HIGHEST_PROTOCOL)
    return print("data named {} has been saved...".format(name))


def anti_norm(data, mean, range):
    """Inverse transformation of output"""
    shape = data.shape
    if len(shape) == 2:
        return data * range + mean
    if len(shape) == 4:
        _data = np.squeeze(data)
        return _data * range + mean


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('logging_iai.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def log(original_function):
    def wrapper_function(*args, **kwargs):
        assert isinstance(args[0], tf.Tensor)
        logger.info('{:s}: Input_shape is  {}'.format(original_function.__name__, args[0].get_shape()))
        out = original_function(*args, **kwargs)
        logger.info('{:s}: Output_shape is  {}'.format(original_function.__name__, out.get_shape()))
        return out
    return wrapper_function

if __name__ == '__main__':
    check_dir('../Test')



