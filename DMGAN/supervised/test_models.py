"""
Author : Nianzu Ethan Zheng
Datetime : 2018-2-2
Place: Shenyang China
Copyright
For Visualization of training process and for Comparision of the different parameters
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from plot import Visualizer
import numpy as np
from sequential.utils import pickle_load, logger
from base_options import BaseConfig
import pylab

vis = Visualizer()
opt = BaseConfig()


def compare_indices(indices_name, sub_number=1, description=None):
    """
    Comparision of different parameters
    Keyword arguments:
        indices_name, sub_number: physical process name and order
        description: the name of plot title

    return a figure save into opt.t_p.
    """
    if indices_name == 'Q':
        if sub_number > 1:
            raise Exception('the number you want don not exist, it should be 0 or 1')
    elif indices_name == 'r':
        if sub_number > 6:
            raise Exception('the number you want don not exist, it should be 0 - 6')
    else:
        Exception('the indices name you want is not valid,try again')

    # obtain file names
    file_names = os.listdir(
        opt.history_test_path
    )
    set = []
    name = []
    # baseline
    path_name = os.path.join(
        opt.history_test_path,
        file_names[0]
    )
    ensemble, _ = pickle_load(path_name)
    set.append(ensemble['actual_' + indices_name][:, sub_number])
    name.append('origin')

    # collection of result
    for file_name in file_names:
        path = os.path.join(opt.history_test_path, file_name)
        ensemble,_ = pickle_load(path)
        set.append(
            ensemble['result_' + indices_name][:, sub_number]
        )
        name.append(file_name)

    # plot
    set = np.array(set).T
    name = [item.replace('.pkl', '') for item in name]
    vis.mtsplot(set, name, 'the comparision of ' + description, opt.history_result_path)
    pylab.close()




if __name__ == '__main__':
    compare_indices(indices_name='Q', sub_number=0, description='Q1')
    compare_indices(indices_name='Q', sub_number=1, description='Q2')
    for number in range(7):
        compare_indices(indices_name='r', sub_number=number, description='R' + str(number))




