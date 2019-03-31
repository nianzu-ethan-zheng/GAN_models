"""
Discription: A Python Progress Meter
Author: Nianzu Ethan Zheng
Date: 2018-1-21
Copyright
"""
import math


def create_lr_schedule(lr_base, decay_rate, decay_epochs, truncated_epoch, mode=None):
    return lambda epoch: _lr_schedule(epoch,  _lr_base=lr_base, _decay_rate=decay_rate,
                                      _decay_epochs=decay_epochs, _truncated_epoch=truncated_epoch, _mode=mode)


def _lr_schedule(_current_epoch, _lr_base=0.002, _decay_rate=0.1, _decay_epochs=500,
                 _truncated_epoch=2000, _mode='constant'):

    if _mode is 'ladder':
        if _current_epoch < _truncated_epoch:
            learning_rate = _lr_base * _decay_rate**math.ceil(_current_epoch/_decay_epochs)
        else:
            learning_rate = _lr_base * _decay_rate**math.ceil(_truncated_epoch/_decay_epochs)
        
    elif _mode is 'exp':  # exponential_decay, exp for shorthand
        if _current_epoch < _truncated_epoch:
            learning_rate = _lr_base * _decay_rate ** (_current_epoch / _decay_epochs)
        else:
            learning_rate = _lr_base * _decay_rate ** (_truncated_epoch / _decay_epochs)
            
    elif _mode is 'constant':
            learning_rate = _lr_base
    else:
        raise Exception('Please select the defined _mode,i.e.,constant')
    return learning_rate

if __name__ == '__main__':
    lr_schedule_c = create_lr_schedule(lr_base=2e-3, decay_rate=0.1, decay_epochs=500,
                                       truncated_epoch=2000, mode="constant")
    lr_schedule_l = create_lr_schedule(lr_base=2e-2, decay_rate=0.1, decay_epochs=500,
                                       truncated_epoch=2000, mode="ladder")
    lr_schedule_e = create_lr_schedule(lr_base=2e-2, decay_rate=0.1, decay_epochs=500,
                                       truncated_epoch=2000, mode="exp")

    import matplotlib.pylab as plt
    import numpy as np
    learning_rate = []
    for epoch in range(3000):
        learning_rate.append([lr_schedule_c(epoch), lr_schedule_l(epoch), lr_schedule_e(epoch)])

    lr = np.array(learning_rate)
    print(lr.shape)
    plt.plot(lr, '-*')
    plt.legend(['constant', 'ladder', 'exponent decay'])
    plt.show()



