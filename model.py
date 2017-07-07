import numpy as np
import tensorflow as tf

class DRAW:
    def __init__(self):
        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            self._build_network()

    def _build_network(
        self,
        with_attention=False,
        training=True,
    ):
        c_prev = tf.placeholder(
            dtype=tf.float32,
            shape=(minibatch_size, input_size),
            name='c_{t-1}',
        )
        h_dec_prev = tf.placeholder(
            dtype=tf.float32,
            shape=(minibatch_size, num_units),
            name='h^{dec}_{t-1}',
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
                name='h^{enc}_{t-1}',
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
                    name='enc_inputs',
                )
                enc_c_state_prev = tf.placeholder(
                    dtype=tf.float32,
                    shape=(minibatch_size, num_units),
                    name='enc_c_state_prev',
                )
                h_enc, enc_state = enc_lstm_cell(
                    inputs,
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
                    name='mu_{t}',
                )

                W_sigma = tf.get_variable(
                    name='W_mu',
                    shape=(num_units, num_zs),
                    initializer=self._get_variable_initializer(),
                )
                sigma = tf.exp(
                    tf.matmul(h_enc, W_sigma),
                    name='sigma_{t}',
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
            )
            
        with tf.variable_scope('decoder'):
            dec_lstm_cell = tf.nn.rnn_cell.LSTMCell(
                num_units=num_units,
                use_peepholes=True,
                forget_bias=1.0,
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

        c = tf.reshape(
            c,
            shape=(minibatch_size, image_size, image_size),
            name='c_{t}'
        )
        h_dec = tf.identity(
            h_dec,
            name='h^{dec}_{t}',
        )

        if training:
            h_enc = tf.identity(
                h_enc,
                name='h^{enc}_{t}',
            )
            with tf.variable_scope('loss'):
                L_x = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=x,
                        logits=c,
                    ),
                    name='L^x',
                )

                L_z_prev = tf.placeholder(
                    dtype=tf.float32,
                    shape=mu.shape,
                    name='L^z_{t-1}'
                )
                sigma_2 = sigma ** 2
                L_z = tf.add(
                    L_z_prev,
                    (mu ** 2 + sigma_2 - tf.log(sigma_2)),
                    name='L^z',
                )



