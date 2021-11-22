from models import convolutional_vae
from configurations import ds_config
from util import dataset, plotter
import time

data_path = '/Users/andreavisentin/ADI/'

# Data creation and load
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path,'Curr_Grundfoss')
ds.dataset_creation()
ds.data_save(ds.name)
ds.data_summary()

vae = convolutional_vae.ConvolutionalVAE()

p = {'first_conv_layer_dim': 64,
     'first_window_size': 7,
     'first_stride': 3,
     'conv_hidden_layers_5': 4,
     'conv_layer_dim_5': 64,
     'window_size_5': 5,
     'conv_additional_layer_5': 0,
     'conv_hidden_layers_2': 2,
     'conv_layer_dim_2': 64,
     'window_size_2': 3,
     'conv_additional_layer_2': 0,
     'dense_hidden_layers': 2,
     'dense_layer_dim': 32,
     'latent_dim': 4,
     'batch_size': 5,
     'epochs': 3,
     'patience': 2,
     'optimizer': 'adam',
     'conv_activation': 'elu',
     'dense_activation': 'elu',
     'lr': 1E-3,
     'decay': 1E-3,
     'conv_kernel_init': 'glorot_uniform',
     'dense_kernel_init': 'he_normal'
     }
# vae.training(ds.X_train, None, None, None, p)
#
# plotter.plot_latent_space(vae, ds.X_test, ds.metadata_test[:,0])