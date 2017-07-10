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

        x_tilde = tf.sigmoid(
            c,
            name='x_tilde',
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
                    shape=(),
                    name='L_z_prev'
                )
                sigma_2 = sigma ** 2
                L_z = tf.add(
                    L_z_prev,
                    tf.reduce_sum(mu ** 2 + sigma_2 - tf.log(sigma_2)),
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

    def get_samples_from_data(self, minibatch_size=1):
        dataset_name=self._config['dataset_name']
        
        if dataset_name == 'MNIST':
            samples = self._mnist_data['x_train'][
                np.random.randint(
                    low=0,
                    high=self._mnist_data['x_size'],
                    size=minibatch_size,
                )
            ]
            samples = samples[:,:,:,np.newaxis]
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
        num_encoding_steps = self._config['num_encoding_steps']

        minibatch_size = self._config['minibatch_size']
        image_size = self._config['image_size']
        input_size = image_size ** 2
        num_units = self._config['num_units']

#        c_0 = np.zeros(
#            shape=(minibatch_size, input_size),
#            dtype=np.float32,
#        )
#        h_enc_0 = np.zeros(
#            shape=(minibatch_size, num_units),
#            dtype=np.float32,
#        )
#        c_enc_state_0 = np.zeros(
#            shape=(minibatch_size, num_units),
#            dtype=np.float32,
#        )
#        h_dec_0 = np.zeros(
#            shape=(minibatch_size, num_units),
#            dtype=np.float32,
#        )
#        c_dec_state_0 = np.zeros(
#            shape=(minibatch_size, num_units),
#            dtype=np.float32,
#        )

        fetches = {}
        for var_name in [
            'h_enc',
            'encoder/c_state',
            'h_dec',
            'decoder/c_state',
            'c',
            'loss/L_x',
            'loss/L_z',
            'loss/L',
        ]:
            fetches[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )
        op_name = 'train/minimize_L'
        fetches[op_name] = self._tf_graph.get_operation_by_name(
            op_name        
        )

        feed_dict = {}
        feed_dict_key = {}
        for var_name in [
            'x',
            'h_enc_prev',
            'encoder/c_state_prev',
            'h_dec_prev',
            'decoder/c_state_prev',
            'c_prev',
            'loss/L_z_prev'
        ]:
            var = feed_dict_key[var_name] = self._tf_graph.get_tensor_by_name(
               var_name + ':0'
            )
            feed_dict[var] = None

        for i in range(num_training_iterations):
            training_samples = self.get_samples_from_data(
                minibatch_size=minibatch_size,
            )
            for k, v in feed_dict.items():
                if k == 'x':
                    feed_dict[k] = training_samples
                else:
                    feed_dict[k] = np.zeros(
                        shape=k.shape.as_list(),
                        dtype=np.float32,
                    )

            for t in range(num_encoding_steps):
                rd = self._tf_session.run(
                    fetches=fetches,
                    feed_dict=feed_dict,
                )

                for var_name in [
                    'h_enc',
                    'encoder/c_state',
                    'h_dec',
                    'decoder/c_state',
                    'c',
                    'loss/L_z',
                ]:
                    var = feed_dict_key[var_name + '_prev']
                    feed_dict[var] = rd[var_name]

            if i % 100 == 0:
                print(
                    'L_x = {:g}, L_z = {:g}, L = {:g}'
                    .format(rd['loss/L_x'], rd['loss/L_z'], rd['loss/L'])
                )

            rd['x'] = training_samples 

        return rd

    def generate_samples(self):
        num_decoding_steps = self._config['num_decoding_steps']

        fetches = {}
        for var_name in [
            'h_dec',
            'decoder/c_state',
            'c',
            'x_tilde',
        ]:
            fetches[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )

        feed_dict = {}
        feed_dict_key = {}
        for var_name in [
            'Q/z',
            'h_dec_prev',
            'decoder/c_state_prev',
            'c_prev',
        ]:
            var = self._tf_graph.get_tensor_by_name(
               var_name + ':0'
            )
            var_shape = var.shape.as_list()
            feed_dict_key[var_name] = var
            if var_name  == 'Q/z':
                feed_dict[var] = np.random.normal(
                    size=var_shape,
                )
            else:
                feed_dict[var] = np.zeros(
                    shape=var_shape,
                    dtype=np.float32,
                )

        rds = [None] * num_decoding_steps

        for t in range(num_decoding_steps):
            rds[t] = self._tf_session.run(
                fetches=fetches,
                feed_dict=feed_dict,
            )

            for var_name in [
                'h_dec',
                'decoder/c_state',
                'c',
            ]:
                var = feed_dict_key[var_name + '_prev']
                feed_dict[var] = rds[t][var_name]

        return rds
