flux = {'batch_size': [20], 'conv_activation': ['relu'], 'conv_additional_layer_2': [1], 'conv_additional_layer_5': [0],
        'conv_hidden_layers_2': [3], 'conv_hidden_layers_5': [2], 'conv_kernel_init': ['glorot_uniform'],
        'conv_layer_dim_2': [64], 'conv_layer_dim_5': [256], 'decay': [0.01], 'dense_activation': ['elu'],
        'dense_hidden_layers': [1], 'dense_kernel_init': ['he_normal'], 'dense_layer_dim': [32], 'epochs': [1000],
        'first_conv_layer_dim': [128], 'first_stride': [3], 'first_window_size': [9], 'latent_dim': [32], 'lr': [0.001],
        'optimizer': ['nadam'], 'patience': [30], 'window_size_2': [3], 'window_size_5': [9]}


def convert_to_talos_dict(d1: dict):
    params = [[x] for x in d1.values]

    return dict(zip(d1.keys(), params))
