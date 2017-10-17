import tensorflow as tf
import tensorflow.contrib.layers as ly
import sequential.LossFunctions as Loss
import sequential.layers
import sequential.learning_rate
import sequential.utility


# config setting include two parts:
# First part is the aee.py method's parameters, which are used for initializing the method
# according to different occasion
# Second part is for model parameters, which are used for build the graph
class Config:
    def __init__(self):
        self.ndim_x = 28 ** 2
        self.ndim_y = 10
        self.ndim_z = 2
        self.distribution_z = 'deterministic'
        self.weight_std = 0.01
        self.weight_initializer = ly.batch_norm
        self.nonlinearity = tf.nn.relu
        self.optimizer_is_adam = True
        self.scale_ratio = 1                             # ratio between reconstruction loss and adversarial loss
        self.batch_size = 100
        self.learning_rate = 0.0001
        self.momentum = 0.5
        self.gradient_clipping_upper = 5
        self.gradient_clipping_lower = -5
        self.weight_decay = 0
        self.device = '/gpu:0'
        self.ckpt_dir = './ckpt_dir'
        self.log_dir = './log_dir'
        self.distribution_sampler = 'gaussian_mixture'
        self.num_types_of_label = 10


class Operation:
    config = Config()

    def optimize(self, loss, variables, optimizer, learning_rate, global_step, is_clipping=True):
        optim = optimizer(learning_rate, beta1=self.config.momentum)
        # Compute Gradients
        gradients_and_vars = optim.compute_gradients(loss, var_list=variables)
        if is_clipping:
            gradients_and_vars = [(grad_var[0] if grad_var[0] is None else
                                   tf.clip_by_value(grad_var[0], self.config.gradient_clipping_lower,
                                                    self.config.gradient_clipping_upper), grad_var[1])
                                  for grad_var in gradients_and_vars]
        return optim.apply_gradients(gradients_and_vars, global_step=global_step)

    def gaussian_noise_layer(self, input_tensor, std=0.3):
        return sequential.layers.gaussian_noise_layer(input_tensor, std=std)

    # loss function based on wasserstein_distance
    def wasserstein_distance(self, f_fake, f_true, for_generator=False):
        return Loss.wasserstein_distance(f_fake, f_true, for_generator=for_generator)

    def euclidean_distance(self, x, x_construction):
        return Loss.euclidean_distance(x, x_construction)

    def jensen_shannon_divergence(self, d_fake, d_true, for_generator=False):
        return Loss.jensen_shannon_divergence(d_fake, d_true, for_generator=for_generator)

    def sigmoid_cross_entropy(self, d_fake, d_true, for_generator=False):
        return Loss.sigmoid_cross_entropy(d_fake, d_true, for_generator=for_generator)

    def softmax_cross_entropy(self, d_fake, d_true, class_fake, class_true, for_generator=False):
        return Loss.softmax_cross_entropy(d_fake, d_true, class_fake, class_true, for_generator=for_generator)

    def constant_learning_rate(self, init=0.0001):
        return sequential.learning_rate.constant(initial_learning_rate=init)

    def ladder_learning_rate(self, current_epoch, init=0.0001):
        return sequential.learning_rate.ladder(current_epoch, initial_learning_rate=init)

    def exponential_decay_learning_rate(self, current_epoch, init=0.002, dr=0.1):
        return sequential.learning_rate.exponential_decay(current_epoch, initial_learning_rate=init, decay_rate=dr)

    # Utils
    def check_dir(self, _dir, is_restart=False):
        return sequential.utility.check_dir(_dir, is_restart=is_restart)
