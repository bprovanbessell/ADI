from util import dataset
import time


class ParametersConfiguration:
    def __init__(self):
        return

    def ConvVaeParameters(self, config_name='Vib_Grundfoss'):
        if config_name == 'vae1':
            p = {'batch_size': 5, 'conv_activation': 'relu', 'conv_additional_layer_2': 1, 'conv_additional_layer_5': 3,
                 'conv_hidden_layers_2': 1, 'conv_hidden_layers_5': 2, 'conv_kernel_init': 'he_normal',
                 'conv_layer_dim_2': 64, 'conv_layer_dim_5': 256, 'decay': 0.001, 'dense_activation': 'elu',
                 'dense_hidden_layers': 1, 'dense_kernel_init': 'he_normal', 'dense_layer_dim': 32, 'epochs': 400,
                 'first_conv_layer_dim': 512, 'first_stride': 3, 'first_window_size': 5, 'latent_dim': 16, 'lr': 0.001,
                 'optimizer': 'adam', 'patience': 30, 'window_size_2': 3, 'window_size_5': 7}
        return p
