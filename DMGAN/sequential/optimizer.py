"""
Discription: A Python Progress Meter
Author: Nianzu Ethan Zheng
Date: 2018-1-21
Copyright
"""
import tensorflow as tf
from sequential.nn_options import NetConfig
config = NetConfig()


def optimize(loss, variables, global_step, learning_rate,
             optimizer=tf.train.AdamOptimizer if config.optimizer_is_adam is True else tf.train.RMSPropOptimizer,
             momentum=config.momentum,
             is_clipping=True,
             lower_bound=config.gradient_clipping_lower,
             upper_bound=config.gradient_clipping_upper):
    """Optimize the loss function with clip weight"""
    optim = optimizer(learning_rate, beta1=momentum)
    # Compute Gradients
    gradients_and_vars = optim.compute_gradients(loss, var_list=variables)
    if is_clipping:
        gradients_and_vars = [(grad_var[0] if grad_var[0] is None else
                               tf.clip_by_value(grad_var[0], lower_bound, upper_bound), grad_var[1])
                              for grad_var in gradients_and_vars]
    return optim.apply_gradients(gradients_and_vars, global_step=global_step)


