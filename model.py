import os
import time
import json

import numpy as np
import tensorflow as tf

LOG_DIR = 'logs'
CHECKPOINT_DIR = 'checkpoints'
CFG_DIR = 'configs'

class DRAW:
    def __init__(
        self,
        config,
#        training=None,
        gpu_memory_fraction=None,
        gpu_memory_allow_growth=True,
    ):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        if not os.path.exists(CFG_DIR):
            os.makedirs(CFG_DIR)

        self._config = config

#        if training is None:
#            raise ValueError('Set training either to be True or False.')
#        else:
#            self._training = training

        self._load_data()

        self._tf_config = tf.ConfigProto()
        self._tf_config.gpu_options.allow_growth = gpu_memory_allow_growth 
        if gpu_memory_fraction is not None:
            self._tf_config.gpu_options.per_process_gpu_memory_fraction = (
                gpu_memory_fraction
            )

        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            self._build_network(
#                training=training,
            )

            self._tf_session = tf.Session(config=self._tf_config)
            self._tf_session.run(tf.global_variables_initializer())

    def _build_network(
        self,
        with_attention=False,
#        training=True,
    ):
        num_time_steps  = self._config['num_time_steps']

        minibatch_size = self._config['minibatch_size']
        image_size = self._config['image_size']
        input_size = image_size ** 2
        num_units = self._config['num_units']
        num_zs = self._config['num_zs']
        
        cs = tf.get_variable(
            name='cs',
            shape=((num_time_steps + 1), minibatch_size, input_size),
            initializer=tf.zeros_initializer(dtype=tf.float32)
        )

        x = tf.placeholder(
            dtype=tf.float32,
            shape=(minibatch_size, input_size),
            name='x',
        )

        with tf.variable_scope('encoder'):
            encoder = tf.nn.rnn_cell.LSTMCell(
                num_units=num_units,
                use_peepholes=True,
                forget_bias=1.0,
            )

            enc_state = encoder.zero_state(
                batch_size=minibatch_size,
                dtype=tf.float32,
            )

        with tf.variable_scope('decoder'):
            decoder = tf.nn.rnn_cell.LSTMCell(
                num_units=num_units,
                use_peepholes=True,
                forget_bias=1.0,
            )

            dec_state = decoder.zero_state(
                batch_size=minibatch_size,
                dtype=tf.float32,
            )

        with tf.variable_scope('Q'):
            W_mu = tf.get_variable(
                name='W_mu',
                shape=(num_units, num_zs),
                initializer=self._get_variable_initializer(),
            )
            W_sigma = tf.get_variable(
                name='W_sigma',
                shape=(num_units, num_zs),
                initializer=self._get_variable_initializer(),
            )
            mu_squared_sum = tf.zeros(
                shape=(),
                dtype=tf.float32,
            )
            sigma_squared_sum = tf.zeros(
                shape=(),
                dtype=tf.float32,
            )
            log_sigma_squared_sum = tf.zeros(
                shape=(),
                dtype=tf.float32,
            )
            N = tf.random_normal(
                shape=(minibatch_size, num_zs),
                dtype=tf.float32,
            )

        with tf.variable_scope('write'):
            if not with_attention: 
                W = tf.get_variable(
                    name='W',
                    shape=(num_units, input_size),
                    initializer=self._get_variable_initializer(),
                )
            else:
                pass

        for t in range(num_time_steps):
            x_hat = x - tf.sigmoid(cs[t])
            if not with_attention:
                r = tf.concat(
                    (x, x_hat),
                    axis=1,
                )
            else:
                pass

            enc_inputs = tf.concat(
                (r, dec_state.h),
                axis=1,
            )

            with tf.variable_scope('encoder') as scope:
                if t > 0:
                    scope.reuse_variables()
                enc_output, enc_state = encoder(
                    enc_inputs,
                    enc_state,
                )

            with tf.variable_scope('Q', reuse=True):
                W_mu = tf.get_variable('W_mu')
                mu = tf.matmul(enc_state.h, W_mu)
                mu_squared_sum += tf.reduce_sum(mu ** 2)

                W_sigma = tf.get_variable('W_sigma')
                sigma = tf.exp(tf.matmul(enc_state.h, W_sigma))
                sigma_squared = sigma ** 2
                sigma_squared_sum += tf.reduce_sum(sigma_squared)
                log_sigma_squared_sum += tf.reduce_sum(
                    tf.log(sigma_squared)
                )

            z = tf.add(
                mu,
                tf.multiply(sigma, N),
                name='z',
            )
                
            with tf.variable_scope('decoder') as scope:
                if t > 0:
                    scope.reuse_variables()
                dec_output, dec_state = decoder(
                    z,
                    dec_state,
                )
                
            with tf.variable_scope('write', reuse=True):
                if not with_attention: 
                    W = tf.get_variable('W')
                    tf.assign(
                        cs[t + 1],
                        cs[t] + tf.matmul(dec_state.h, W),
                    )
                else:
                    pass

        # End of RNN rollout.

        # XXX
        xs_tilde = tf.sigmoid(
            cs,
            name='xs_tilde',
        )
        x_tilde = tf.sigmoid(
            cs[num_time_steps],
            name='x_tilde',
        )

        with tf.variable_scope('loss'):
            L_x = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=x,
                    logits=cs[num_time_steps],
                ),
                name='L_x',
            )

            L_z = tf.identity(
                (mu_squared_sum + sigma_squared_sum - log_sigma_squared_sum),
                name='L_z',
            )

            L = tf.add(
                L_x, L_z,
                name='L'
            )

        with tf.variable_scope('train'):
            adam = tf.train.AdamOptimizer(**self._config['adam'])

            train_op = adam.minimize(
                loss=L,
                name='minimize_L',
            )

            train_op = tf.group(
                adam.minimize(loss=L_x),
                adam.minimize(loss=L_z),
                name='minimize_losses',
            )

    def _get_variable_initializer(self):
        return tf.truncated_normal_initializer(
            **self._config['variable_initializer']
        )

    def _load_data(self):
        dataset_name=self._config['dataset_name']

        if dataset_name == 'MNIST':
            ((x_train, y_train),
             (x_test, y_test)) = tf.contrib.keras.datasets.mnist.load_data()
            self._mnist_data = {
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
                'x_size': len(x_train),
                'y_size': len(y_train),
            }
            data = np.concatenate((x_train, x_test))
            self._data = np.array(
                (data / np.iinfo(np.uint8).max),
                dtype=np.float32,
            )

    def get_samples_from_data(self, minibatch_size=None):
        dataset_name=self._config['dataset_name']
        if minibatch_size is None:
            minibatch_size = self._config['minibatch_size']
        image_size = self._config['image_size']
        input_size = image_size ** 2
        
        if dataset_name == 'MNIST':
            samples = self._mnist_data['x_train'][
                np.random.randint(
                    low=0,
                    high=self._mnist_data['x_size'],
                    size=minibatch_size,
                )
            ]
            samples = samples.reshape((minibatch_size, input_size))
        elif dataset_name == 'SVHN':
            samples = self._data[
                np.random.randint(
                    low=0,
                    high=len(self._data),
                    size=minibatch_size,
                )
            ]
        else:
            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))

        return samples

    def train(self):
        num_training_iterations = self._config['num_training_iterations']
        display_iterations = num_training_iterations // 100

        minibatch_size = self._config['minibatch_size']
        image_size = self._config['image_size']
        input_size = image_size ** 2
        num_units = self._config['num_units']

        fetches = {}
        for var_name in [
            'cs',
            'xs_tilde',
            'x_tilde',
            'loss/L_x',
            'loss/L_z',
            'loss/L',
        ]:
            fetches[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )

        op_name = 'train/minimize_L'
#        op_name = 'train/minimize_losses'
        fetches['train_op'] = self._tf_graph.get_operation_by_name(op_name)

        x = self._tf_graph.get_tensor_by_name('x:0')

        for i in range(num_training_iterations):
            training_samples = self.get_samples_from_data(
                minibatch_size=minibatch_size,
            )
            feed_dict = {x: training_samples}

            rd = self._tf_session.run(
                fetches=fetches,
                feed_dict=feed_dict,
            )
            rd['x'] = training_samples 

            if i % display_iterations == 0:
                print(
                    'L_x = {:g}, L_z = {:g}, L = {:g}'
                    .format(rd['loss/L_x'], rd['loss/L_z'], rd['loss/L'])
                )

        return rd

    def generate_samples(self):

        fetches = {}
        for var_name in [
            'cs',
            'xs_tilde',
            'x_tilde',
        ]:
            fetches[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )

        z = self._tf_graph.get_tensor_by_name('z:0')
        feed_dict = {
            z: np.random.normal(z.shape.as_list()),
        }

        rd = self._tf_session.run(
            fetches=fetches,
            feed_dict=feed_dict,
        )

        return rd
