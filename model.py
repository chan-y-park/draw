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
        training=None,
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

        if training is None:
            raise ValueError('Set training either to be True or False.')
        else:
            self._training = training

        self._load_data()

        self._tf_config = tf.ConfigProto()
        self._tf_config.gpu_options.allow_growth = gpu_memory_allow_growth 
        if gpu_memory_fraction is not None:
            self._tf_config.gpu_options.per_process_gpu_memory_fraction = (
                gpu_memory_fraction
            )

        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            self._build_network(training=training)

            self._tf_session = tf.Session(config=self._tf_config)
            self._tf_session.run(tf.global_variables_initializer())

    def _build_network(
        self,
        with_attention=False,
        training=True,
    ):
        minibatch_size = self._config['minibatch_size']
        image_size = self._config['image_size']
        input_size = image_size ** 2
        num_units = self._config['num_units']
        num_zs = self._config['num_zs']
        
        c_prev = tf.placeholder(
            dtype=tf.float32,
            shape=(minibatch_size, input_size),
            name='c_prev',
        )
        h_dec_prev = tf.placeholder(
            dtype=tf.float32,
            shape=(minibatch_size, num_units),
            name='h_dec_prev',
        )

        if training:
            x = tf.placeholder(
                dtype=tf.float32,
                shape=(minibatch_size, input_size),
                name='x',
            )
            h_enc_prev = tf.placeholder(
                dtype=tf.float32,
                shape=(minibatch_size, num_units),
                name='h_enc_prev',
            )

            with tf.variable_scope('read'):
                x_hat = x - tf.sigmoid(c_prev)
                if not with_attention:
                    r = tf.concat(
                        (x, x_hat),
                        axis=1,
                        name='r',
                    )
                else:
                    pass

            with tf.variable_scope('encoder'):
                enc_lstm_cell = tf.nn.rnn_cell.LSTMCell(
                    num_units=num_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                )
                enc_inputs = tf.concat(
                    (x, x_hat, h_dec_prev),
                    axis=1,
                    name='inputs',
                )
                enc_c_state_prev = tf.placeholder(
                    dtype=tf.float32,
                    shape=(minibatch_size, num_units),
                    name='c_state_prev',
                )
                h_enc, enc_state = enc_lstm_cell(
                    enc_inputs,
                    (enc_c_state_prev, h_enc_prev),
                )
                enc_c_state = tf.identity(
                    enc_state.c,
                    name='c_state',
                )

            with tf.variable_scope('Q'):
                W_mu = tf.get_variable(
                    name='W_mu',
                    shape=(num_units, num_zs),
                    initializer=self._get_variable_initializer(),
                )
                mu = tf.matmul(
                    h_enc, W_mu,
                    name='mu',
                )

                W_sigma = tf.get_variable(
                    name='W_sigma',
                    shape=(num_units, num_zs),
                    initializer=self._get_variable_initializer(),
                )
                sigma = tf.exp(
                    tf.matmul(h_enc, W_sigma),
                    name='sigma',
                )

                N = tf.random_normal(
                    shape=(minibatch_size, num_zs),
                    dtype=tf.float32,
                )

                z = tf.add(
                    mu,
                    tf.multiply(sigma, N),
                    name='z',
                )
        else:
            z = tf.random_normal(
                shape=(minibatch_size, num_zs),
                name='z',
            )
            
        with tf.variable_scope('decoder'):
            dec_lstm_cell = tf.nn.rnn_cell.LSTMCell(
                num_units=num_units,
                use_peepholes=True,
                forget_bias=1.0,
            )
            dec_c_state_prev = tf.placeholder(
                dtype=tf.float32,
                shape=(minibatch_size, num_units),
                name='c_state_prev',
            )
            h_dec, dec_state = dec_lstm_cell(
                z,
                (dec_c_state_prev, h_dec_prev),
            )
            dec_c_state = tf.identity(
                dec_state.c,
                name='c_state',
            )
            
        with tf.variable_scope('write'):
            if not with_attention: 
                W = tf.get_variable(
                    name='W',
                    shape=(num_units, input_size),
                    initializer=self._get_variable_initializer(),
                )
                c = c_prev + tf.matmul(h_dec, W)
            else:
                pass

        c = tf.identity(
            c,
            name='c'
        )
        h_dec = tf.identity(
            h_dec,
            name='h_dec',
        )

        if training:
            h_enc = tf.identity(
                h_enc,
                name='h_enc',
            )
            with tf.variable_scope('loss'):
                L_x = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=x,
                        logits=c,
                    ),
                    name='L_x',
                )

                L_z_prev = tf.placeholder(
                    dtype=tf.float32,
                    shape=mu.shape,
                    name='L_z_prev'
                )
                sigma_2 = sigma ** 2
                L_z = tf.add(
                    L_z_prev,
                    (mu ** 2 + sigma_2 - tf.log(sigma_2)),
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

    def train(self, num_training_iterations=1):
        fetches = {
            'c': self._tf_graph.get_tensor_by_name(
                'c'
            ),
        }
        feed_dict = {
            self._tf_graph.get_tensor_by_name(
                'x'
            ): self._get_sample_from_data()
        }
