import os
import time
import json
import threading

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
        save_path=None,
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
        self._tf_coordinator = tf.train.Coordinator()

        self._tf_config = tf.ConfigProto()
        self._tf_config.gpu_options.allow_growth = gpu_memory_allow_growth 
        if gpu_memory_fraction is not None:
            self._tf_config.gpu_options.per_process_gpu_memory_fraction = (
                gpu_memory_fraction
            )

        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            if self._training:
                with tf.variable_scope('input_queue'):
                    self._build_input_queue()

            self._build_network()

            if self._training:
                with tf.variable_scope('summary'):
                    self._build_summary_ops()

            self._tf_session = tf.Session(config=self._tf_config)
            self._tf_session.run(tf.global_variables_initializer())

            self._tf_saver = tf.train.Saver()
            if save_path is not None:
                self._tf_saver.restore(self._tf_session, save_path)
                self._iter = get_step_from_checkpoint(save_path)
            else:
                self._iter = None

    def _load_data(self):
        dataset_name=self._config['dataset_name']

        if dataset_name == 'mnist' or dataset_name == 'svhn':
            mnist_data = np.load(
                'datasets/{}.npz'
                .format(dataset_name)
            )
            data = np.concatenate(
                (mnist_data['x_train'], mnist_data['x_test'])
            )
            self._data = np.array(
                (data / np.iinfo(np.uint8).max),
                dtype=np.float32,
            )
        else:
            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))

        image_size = self._config['image_size']
        if dataset_name == 'mnist':
            self._config['input_size'] = image_size ** 2
        elif dataset_name == 'svhn':
            self._config['input_size'] = (image_size ** 2) * 3

    def _build_input_queue(self):
        minibatch_size = self._config['minibatch_size']
        input_size = self._config['input_size']

        queue_inputs = tf.placeholder(
            dtype=tf.float32,
            shape=(None, input_size),
            name='inputs',
        )

        queue_capacity = 2 * minibatch_size

        queue = tf.FIFOQueue(
            capacity=queue_capacity,
            dtypes=[tf.float32],
            shapes=[(input_size)],
            name='real_image_queue',
        )

        close_op = queue.close(
            cancel_pending_enqueues=True,
            name='close_op',
        )

        enqueue_op = queue.enqueue_many(
            queue_inputs,
            name='enqueue_op',
        )

        dequeued_tensors = queue.dequeue_many(
            minibatch_size,
            name='dequeued_tensors',
        )

        size_op = queue.size(
            name='size',
        )

    def _build_network(
        self,
        with_attention=False,
        use_lstm_block_cell=True,
    ):
        training = self._training
        num_time_steps  = self._config['num_time_steps']

        minibatch_size = self._config['minibatch_size']
        input_size = self._config['input_size']
        num_units = self._config['num_units']
        num_zs = self._config['num_zs']
        
        c = tf.get_variable(
            name='c',
            shape=(minibatch_size, input_size),
            initializer=tf.zeros_initializer(dtype=tf.float32)
        )

        lstm_kwargs = {
            'num_units': num_units,
            'forget_bias': 1.0,
        }
        if use_lstm_block_cell:
            tf_lstm_cell = tf.contrib.rnn.LSTMBlockCell
            lstm_kwargs['use_peephole'] = True
        else:
            tf_lstm_cell = tf.nn.rnn_cell.LSTMCell
            lstm_kwargs['use_peepholes'] = True


        if training:
            x = tf.identity(
                self._tf_graph.get_tensor_by_name(
                    'input_queue/dequeued_tensors:0'
                ),
                name='x',
            )
            with tf.variable_scope('encoder'):
                encoder = tf_lstm_cell(**lstm_kwargs)

                enc_state = encoder.zero_state(
                    batch_size=minibatch_size,
                    dtype=tf.float32,
                )

            with tf.variable_scope('Q'):
                W_mu = tf.get_variable(
                    name='W_mu',
                    shape=(num_units, num_zs),
                    initializer=self._get_variable_initializer(),
                )
                b_mu = tf.get_variable(
                    name='b_mu',
                    shape=(minibatch_size, num_zs),
                    initializer=self._get_variable_initializer(),
                )
                W_sigma = tf.get_variable(
                    name='W_sigma',
                    shape=(num_units, num_zs),
                    initializer=self._get_variable_initializer(),
                )
                b_sigma = tf.get_variable(
                    name='b_sigma',
                    shape=(minibatch_size, num_zs),
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
        else:
            cs = [None] * num_time_steps

        with tf.variable_scope('decoder'):
            decoder = tf_lstm_cell(**lstm_kwargs)

            dec_state = decoder.zero_state(
                batch_size=minibatch_size,
                dtype=tf.float32,
            )

        with tf.variable_scope('write'):
            if not with_attention: 
                W = tf.get_variable(
                    name='W',
                    shape=(num_units, input_size),
                    initializer=self._get_variable_initializer(),
                )
                b = tf.get_variable(
                    name='b',
                    shape=(minibatch_size, input_size),
                    initializer=self._get_variable_initializer(),
                )
            else:
                pass

        for t in range(num_time_steps):
            if training:
                x_hat = x - tf.sigmoid(c)
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
                    b_mu = tf.get_variable('b_mu')
                    mu = tf.matmul(enc_state.h, W_mu) + b_mu
                    mu_squared_sum += tf.reduce_sum(mu ** 2)

                    W_sigma = tf.get_variable('W_sigma')
                    b_sigma = tf.get_variable('b_sigma')
                    sigma = tf.exp(tf.matmul(enc_state.h, W_sigma) + b_sigma)
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
            else:
                z = tf.random_normal(
                    shape=(minibatch_size, num_zs),
                    dtype=tf.float32,
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
                    b = tf.get_variable('b')
                    c += tf.matmul(dec_state.h, W) + b
                else:
                    pass

            if not training:
                cs[t] = c

        # End of RNN rollout.

        x_tilde = tf.sigmoid(
            c,
            name='x_tilde',
        )

        if training:
            with tf.variable_scope('loss'):
                L_x = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=x,
                        logits=c,
                    ),
                    name='L_x',
                )

                L_z = tf.identity(
                    (mu_squared_sum + sigma_squared_sum
                     - log_sigma_squared_sum),
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
        else:
            cs = tf.stack(
                cs,
                name='cs',
            )
            xs_tilde = tf.sigmoid(
                cs,
                name='xs_tilde',
            )

    def _get_variable_initializer(self):
        return tf.truncated_normal_initializer(
            **self._config['variable_initializer']
        )

    def _build_summary_ops(self):
        minibatch_size = self._config['minibatch_size']
        image_size = self._config['image_size']

        summaries = []
        with tf.variable_scope('loss'):
            for name in ['L_x', 'L_z', 'L']:
                summaries.append(
                    tf.summary.scalar(
                        name=name,
                        tensor=self._tf_graph.get_tensor_by_name(
                            'loss/{}:0'.format(name)
                        ),
                    )
                )

        with tf.variable_scope('image'):
            x = tf.reshape(
                self._tf_graph.get_tensor_by_name('x:0'),
                shape=(minibatch_size, image_size, image_size, -1)
            )
            summaries.append(
                tf.summary.image(
                    name='input_images',
                    tensor=x,
                )
            )
            x_tilde = tf.reshape(
                self._tf_graph.get_tensor_by_name('x_tilde:0'),
                shape=(minibatch_size, image_size, image_size, -1)
            )
            summaries.append(
                tf.summary.image(
                    name='generated_images',
                    tensor=x_tilde,
                )
            )

        summary_op = tf.summary.merge(
            summaries,
            name='merged',
        )

    def get_samples_from_data(self, minibatch_size=None):
        dataset_name=self._config['dataset_name']
        if minibatch_size is None:
            minibatch_size = self._config['minibatch_size']
        image_size = self._config['image_size']
        input_size = self._config['input_size']

        samples = self._data[
            np.random.randint(
                low=0,
                high=len(self._data),
                size=minibatch_size,
            )
        ]
#        samples = samples.reshape((minibatch_size, input_size))
        return samples

    def _enqueue_thread(self):
        minibatch_size = self._config['minibatch_size']
        input_size = self._config['input_size']

        num_data = len(self._data)
        i = 0
        num_elements = minibatch_size

        enqueue_op = self._tf_graph.get_operation_by_name(
            'input_queue/enqueue_op'
        )
        queue_inputs = self._tf_graph.get_tensor_by_name(
            'input_queue/inputs:0' 
        )

        np.random.shuffle(self._data)

        while not self._tf_coordinator.should_stop():
            if (i + num_elements) <= num_data:
                data_to_enqueue = self._data[i:(i + num_elements)]
                i += num_elements
            else:
                data_to_enqueue = self._data[i:]
                i = num_elements - (num_data - i)
                data_to_enqueue = np.concatenate(
                    (data_to_enqueue, self._data[:i]),
                )
                np.random.shuffle(self._data)
            data_to_enqueue = data_to_enqueue.reshape(
                (minibatch_size, input_size),
            )
            try: 
                self._tf_session.run(
                    enqueue_op,
                    feed_dict={queue_inputs: data_to_enqueue}
                )
            except tf.errors.CancelledError:
#                print('Input queue closed.')
                pass

    def train(
        self,
        run_name=None,
        max_num_iters=None,
        additional_num_iters=None,
    ):
        if not self._training:
            raise RuntimeError

        if run_name is None:
            run_name = (
                '{:02}{:02}_{:02}{:02}{:02}'.format(*time.localtime()[1:6])
            )

        summary_writer = tf.summary.FileWriter(
            logdir='{}/{}'.format(LOG_DIR, run_name),
            graph=self._tf_graph,
        )

        if self._iter is None:
            self._iter = 1
        if max_num_iters is not None:
            self._config['num_training_iterations'] = max_num_iters
        if additional_num_iters is not None:
           self._config['num_training_iterations'] += additional_num_iters

        num_training_iterations = self._config['num_training_iterations']
        display_iterations = num_training_iterations // 100
        save_iterations = num_training_iterations // 10

#        minibatch_size = self._config['minibatch_size']
#        image_size = self._config['image_size']
#        input_size = image_size ** 2
#        num_units = self._config['num_units']

        queue_threads = [threading.Thread(target=self._enqueue_thread)]
        for t in queue_threads:
            t.start()

        fetches = {}
        for var_name in [
            'x',
            'x_tilde',
            'loss/L_x',
            'loss/L_z',
            'loss/L',
            'summary/merged/merged',
        ]:
            fetches[var_name] = self._tf_graph.get_tensor_by_name(
                var_name + ':0'
            )

        for op_name in [
#            'train/minimize_L'
            'train/minimize_losses',
        ]:
            fetches[op_name] = self._tf_graph.get_operation_by_name(op_name)

        try:
            for i in range(self._iter, num_training_iterations + 1):
                if self._tf_coordinator.should_stop():
                    break

                rd = self._tf_session.run(
                    fetches=fetches,
                )

                summary_writer.add_summary(
                    summary=rd['summary/merged/merged'],
                    global_step=i,
                )

                if i % display_iterations == 0:
                    print(
                        '{:g}% : L_x = {:g}, L_z = {:g}, L = {:g}'
                        .format(
                            (i / num_training_iterations * 100),
                            rd['loss/L_x'], rd['loss/L_z'], rd['loss/L'],
                        ),
                    )
                if (
                    i % save_iterations == 0
                    or i == num_training_iterations
                ):
                    save_path = self._tf_saver.save(
                        self._tf_session,
                        'checkpoints/{}'.format(run_name),
                        i,
                    )
                    print('checkpoint saved at {}'.format(save_path))

            # End of iteration for-loop.

        except tf.errors.OutOfRangeError:
            raise RuntimeError

        finally:
            self._tf_coordinator.request_stop()
            self._tf_session.run(
                self._tf_graph.get_operation_by_name(
                    'input_queue/close_op'
                )
            )

        self._tf_coordinator.join(queue_threads)

        with open('{}/{}'.format(CFG_DIR, run_name), 'w') as fp:
            json.dump(self._config, fp)

        summary_writer.close()

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

        rd = self._tf_session.run(
            fetches=fetches,
        )

        return rd

def get_step_from_checkpoint(save_path):
    return int(save_path.split('-')[-1])
