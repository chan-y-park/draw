mnist_without_attention = {
    'dataset_name': 'MNIST',
#    'minibatch_size': 100,
    'minibatch_size': 3,
    'image_size': 28,
    'num_units': 256,
    'num_zs': 100,
    'adam': {
#        'learning_rate': 0.0002,
        'learning_rate': 0.001,
#        'beta1': 0.5,
        'beta1': 0.9,
    },
    'variable_initializer': {'mean': 0, 'stddev': 0.02},
#    'num_training_iterations': 1000,
    'num_training_iterations': 100,
    'num_time_steps': 10,
}
