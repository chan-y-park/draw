mnist_without_attention = {
    'dataset_name': 'mnist',
    'minibatch_size': 200,
    'image_size': 28,
    'num_units': 256,
    'num_zs': 100,
    'adam': {
        'learning_rate': 0.001,
        'beta1': 0.9,
    },
    'variable_initializer': {'mean': 0, 'stddev': 0.02},
    'num_training_iterations': 10 ** 3,
    'num_time_steps': 10,
}

svhn_without_attention = {
    'dataset_name': 'svhn',
    'minibatch_size': 200,
    'image_size': 32,
    'num_units': 800,
    'num_zs': 1000,
    'adam': {
        'learning_rate': 0.001,
        'beta1': 0.9,
    },
    'variable_initializer': {'mean': 0, 'stddev': 0.02},
    'num_training_iterations': 10 ** 4,
    'num_time_steps': 10,
}
