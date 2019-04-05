import tensorflow as tf
import os
import shutil
import time
from datetime import timedelta
import numpy as np
from collections import defaultdict
import _pickle as pickle

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class BasicComponent:
    """ Basic Component provide some preliminary method """

    def conv2d(self, _input, out_features, kernel_size,
               strides=None, padding='SAME'):
        if strides is None:
            strides = [1, 1, 1, 1]
        in_features = int(_input.get_shape()[-1])
        kernel_size = self.weight_variable_msra(
            [1, kernel_size, in_features, out_features],
            name='kernel'
        )
        out_features = tf.nn.conv2d(_input, kernel_size, strides=strides, padding=padding)
        return out_features

    def avg_pool(self, _input, k):
        ksize = [1, 1, k, 1]
        strides = [1, 1, k, 1]
        padding = 'SAME'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    is_training = False

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training)
        return output

    keep_prob = 1

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, keep_prob=self.keep_prob),
                lambda: _input,
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer()
        )

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer()
        )

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name=name, initializer=initial)


class DenseNet(BasicComponent):
    def __init__(self, data_provider=None, growth_rate=12, depth=40,
                 total_blocks=3,
                 keep_prob=0.8,
                 weight_decay=1E-4,
                 momentum=0.9,
                 model_type='DenseNet',
                 dataset='C10',
                 should_save_logs=True,
                 should_save_model=True,
                 renew_logs=False,
                 reduction=1,
                 bc_mode=False,
                 run_from_checkpoint=False,
                 **kwargs):
        '''
        :param data_provider:   Class, that have all required dataset
        :param growth_rate:  'int', the number of the feature maps that the layer in dense block output
        :param depth:  'int', how many layers the network has
        :param total_blocks: 'int', how many dense block the network has
        :param keep_prob:  'float' ,keep probability for dropout, If keep_prob = 1, the dropout will be disables
        :param weight_decay: ' float', weigh decay for L2 loss , value in paper =  1e-4
        :param momentum:  'float', momentum for optimizer
        :param model_type: 'str' , 'DenseNet' or 'DenseNet-BC'
        :param dataset:  'str', dataset name
        :param should_save_logs:  'bool', should logs be saved or not
        :param should_save_model: 'bool'
        :param renew_logs: 'bool', remove the previous logs for current model
        :param reduction: 'float', reduction Theta at transition layer for DenseNet with bottleneck layers
        :param bc_mode: 'bool' , should we use bottleneck layers and features reduction or not
        '''

        self.data_provider = data_provider
        self.data_shape = (1, 700, 1)  # data_provider.data_shape
        self.n_classes = 1
        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print('Build %s model with %d blocks,'
                  '%d composite layers each' % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print('Build %s model with %d blocks,'
                  '%d bottleneck layers and %d composite layers each' % (
                      model_type, self.total_blocks, self.layers_per_block, self.layers_per_block))
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0
        self.run_from_checkpoint = run_from_checkpoint
        if self.run_from_checkpoint:
            if not os.path.exists(self.save_path):
                raise Exception(
                    'There is not a checkpoint to restore,'
                    'please reset run_from_checkpoint to False')

        self._define_input()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """
        Load from checkpoint or start a new session
        """
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver(
            max_to_keep=2)
        if self.run_from_checkpoint:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))
            self.logs_writer = self.load_logs_writer()
            print('Successfully load model from save path %s' % self.save_path)

        else:
            self.sess.run(tf.global_variables_initializer())
            self.logs_writer = defaultdict(list)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total training params: %.1fM' % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model_chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return '{}_growth_rate={}_depth={}'.format(
            self.model_type, self.growth_rate, self.depth
        )

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
            print('Successfully load model from save path %s' % self.save_path)
        except Exception as e:
            raise IOError('Failed to load model'
                          'from save path %s' % self.save_path)

    def load_logs_writer(self):
        try:
            logs_writer = pickle.load(open(
                self.logs_path + '%s.pkl' % self.model_identifier, 'rb'
            ))
        except Exception:
            raise IOError('Failed to load logs_writer'
                          'from save_path %s' % self.save_path)
        return logs_writer

    def log_loss_accuracy(self, loss, epoch, _dest, prefix, should_print=True):
        if should_print:
            print('%s: %.4f' % (_dest, loss))
        self.logs_writer[prefix].append([epoch, loss])

    def _define_input(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images'
        )
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels'
        )
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate'
        )
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        """
        To be consist of Functions:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope('composite_function'):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size
            )
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope('bottleneck'):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(output, out_features=inter_features, kernel_size=1, padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and  after concatenate
        input with output from composite function
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3
            )
        output = tf.concat(axis=3, values=(_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_L inter layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope('layer_%d' % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average pooling"""
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1
        )
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    def transition_layer_to_classes(self, _input):
        """ This is last transition to get probability by classes, it perform:
        - batch_normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W'
        )
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first output_features
        with tf.variable_scope('Initial_convolution'):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)
        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output)
        self.prediction = tf.nn.tanh(logits)
        # Losses
        self.mean_square_error = tf.reduce_mean(0.5 * tf.square(self.prediction))
        L2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        # optimizer and train step
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=self.momentum)
        self.train_step = optimizer.minimize(
            self.mean_square_error + L2_loss * self.weight_decay)

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 40, '\n')
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate /= 10
                print("Decrease learning rate, New lr = %f" % learning_rate)

            print("Training")
            loss = self.train_one_epoch(self.data_provider.train,
                                             batch_size, learning_rate)

            if self.should_save_logs:
                self.log_loss_accuracy(loss, epoch, _dest='mean_square_error_train', prefix='train_loss')

            if train_params.get('validation_set', False):
                print("Validation...")
                loss, correlation_test = self.test(
                    self.data_provider.validation, batch_size)
                _, correlation_train = self.test(
                    self.data_provider.train, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, epoch, _dest='mean_square_error_validation', prefix='valid_loss')
                    self.log_loss_accuracy(correlation_test, epoch, _dest='correlation_validation', prefix='valid_r')
                    self.log_loss_accuracy(correlation_train, epoch, _dest='correlation_train', prefix='train_r')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

        total_training_time = time.time() - total_start_time
        print("\nTotal training time:%s" % str(timedelta(seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate):
        num_examples = data.num_examples
        total_loss = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True
            }
            fetches = [self.train_step, self.mean_square_error]
            _, loss = self.sess.run(fetches, feed_dict=feed_dict)
            total_loss.append(loss)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(loss, self.batches_step, prefix='train_per_batch', _dest='train_per_batch',should_print=False)

        mean_loss = np.mean(total_loss)
        return mean_loss

    def test(self, data, batch_size):
        # training process correlation
        batch_num = data.num_examples // batch_size
        feed_dict = {
            self.images: data.images,
            self.labels: data.labels,
            self.is_training: False,
        }
        label_pre, loss = self.sess.run(
            [self.prediction, self.mean_square_error],
            feed_dict=feed_dict)
        from sklearn.metrics import r2_score
        correlation = r2_score(y_true=data.labels, y_pred=label_pre)
        return loss / batch_num, correlation
