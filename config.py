mnist_without_attention = {
    'dataset_name': 'MNIST',
    'minibatch_size': 100,
    'image_size': 28,
    'num_units': 256,
    'num_zs': 100,
    'adam': {
        'learning_rate': 0.0002,
        'beta1': 0.5,
    },
    'variable_initializer': {'mean': 0, 'stddev': 0.02},
    'num_training_iterations': 1000,
    'num_encoding_steps': 10,
    'num_decoding_steps': 10,
}
