flux = {'batch_size': [20], 'conv_activation': ['relu'], 'conv_additional_layer_2': [1], 'conv_additional_layer_5': [0],
        'conv_hidden_layers_2': [3], 'conv_hidden_layers_5': [2], 'conv_kernel_init': ['glorot_uniform'],
        'conv_layer_dim_2': [64], 'conv_layer_dim_5': [256], 'decay': [0.01], 'dense_activation': ['elu'],
        'dense_hidden_layers': [1], 'dense_kernel_init': ['he_normal'], 'dense_layer_dim': [32], 'epochs': [1000],
        'first_conv_layer_dim': [128], 'first_stride': [3], 'first_window_size': [9], 'latent_dim': [32], 'lr': [0.001],
        'optimizer': ['nadam'], 'patience': [30], 'window_size_2': [3], 'window_size_5': [9]}


def convert_to_talos_dict(d1: dict):
    params = [[x] for x in d1.values()]

    return dict(zip(d1.keys(), params))


vib = {'batch_size': [20], 'conv_activation': ['elu'], 'conv_additional_layer_2': [1], 'conv_additional_layer_5': [2],
       'conv_hidden_layers_2': [2], 'conv_hidden_layers_5': [3], 'conv_kernel_init': ['glorot_uniform'],
       'conv_layer_dim_2': [256], 'conv_layer_dim_5': [256], 'decay': [0.0001], 'dense_activation': ['selu'],
       'dense_hidden_layers': [1], 'dense_kernel_init': ['he_normal'], 'dense_layer_dim': [64], 'epochs': [1000],
       'first_conv_layer_dim': [512], 'first_stride': [1], 'first_window_size': [5], 'latent_dim': [8], 'lr': [0.01],
       'optimizer': ['adam'], 'patience': [30], 'window_size_2': [9], 'window_size_5': [3]}

# {'batch_size': 20, 'conv_activation': 'elu', 'conv_additional_layer_2': 1, 'conv_additional_layer_5': 2,
#    'conv_hidden_layers_2': 2, 'conv_hidden_layers_5': 3, 'conv_kernel_init': 'glorot_uniform',
#    'conv_layer_dim_2': 256, 'conv_layer_dim_5': 256, 'decay': 0.0001, 'dense_activation': 'selu',
#    'dense_hidden_layers': 1, 'dense_kernel_init': 'he_normal', 'dense_layer_dim': 64, 'epochs': 1000,
#    'first_conv_layer_dim': 512, 'first_stride': 1, 'first_window_size': 5, 'latent_dim': 8, 'lr': 0.01,
#    'optimizer': 'adam', 'patience': 30, 'window_size_2': 9, 'window_size_5': 3}

{'batch_size': [20], 'conv_activation': ['relu'], 'conv_additional_layer_2': [1], 'conv_additional_layer_5': [0],
 'conv_hidden_layers_2': [3], 'conv_hidden_layers_5': [3], 'conv_kernel_init': ['glorot_uniform'],
 'conv_layer_dim_2': [128], 'conv_layer_dim_5': [64], 'decay': [0.001], 'dense_activation': ['selu'],
 'dense_hidden_layers': [1], 'dense_kernel_init': ['he_normal'], 'dense_layer_dim': [256], 'epochs': [1000],
 'first_conv_layer_dim': [512], 'first_stride': [1], 'first_window_size': [1], 'latent_dim': [32], 'lr': [0.001],
 'optimizer': ['adam'], 'patience': [30], 'window_size_2': [9], 'window_size_5': [9]}

# curr = {'batch_size': 20, 'conv_activation': 'relu', 'conv_additional_layer_2': 1, 'conv_additional_layer_5': 0,
#         'conv_hidden_layers_2': 3, 'conv_hidden_layers_5': 3, 'conv_kernel_init': 'glorot_uniform',
#         'conv_layer_dim_2': 128, 'conv_layer_dim_5': 64, 'decay': 0.001, 'dense_activation': 'selu',
#         'dense_hidden_layers': 1, 'dense_kernel_init': 'he_normal', 'dense_layer_dim': 256, 'epochs': 1000,
#         'first_conv_layer_dim': 512, 'first_stride': 1, 'first_window_size': 1, 'latent_dim': 32, 'lr': 0.001,
#         'optimizer': 'adam', 'patience': 30, 'window_size_2': 9, 'window_size_5': 9}
