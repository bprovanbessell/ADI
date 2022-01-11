from models import convolutional_vae
from configurations import ds_config
from util import dataset
import os
import talos
import pickle

data_path = '/Users/andreavisentin/ADI/data_tm/'
data_path = '/home/avisentin/data/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
experiment_name = 'curr_sept_oct_gcl_error'

# Data creation and load
# Make new dataset configuration for all of these tests
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, experiment_name)
# ds = ds.data_load(ds.name)
ds.dataset_creation_influx()
ds.data_summary()

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

p = {'batch_size': [5], 'conv_activation': ['relu'], 'conv_additional_layer_2': [1], 'conv_additional_layer_5': [3],
     'conv_hidden_layers_2': [1], 'conv_hidden_layers_5': [2], 'conv_kernel_init': ['he_normal'],
     'conv_layer_dim_2': [64], 'conv_layer_dim_5': [256], 'decay': [0.001], 'dense_activation': ['elu'],
     'dense_hidden_layers': [1], 'dense_kernel_init': ['he_normal'], 'dense_layer_dim': [32], 'epochs': [400],
     'first_conv_layer_dim': [512], 'first_stride': [3], 'first_window_size': [5], 'latent_dim': [16], 'lr': [0.001],
     'optimizer': ['adam'], 'patience': [30], 'window_size_2': [3], 'window_size_5': [7]}

parameter_list = {'first_conv_layer_dim': [512],
                  'first_window_size': [5],
                  'first_stride': [3],
                  'conv_hidden_layers_5': [2],
                  'conv_layer_dim_5': [256],
                  'window_size_5': [7],
                  'conv_additional_layer_5': [3],
                  'conv_hidden_layers_2': [1],
                  'conv_layer_dim_2': [64],
                  'window_size_2': [3],
                  'conv_additional_layer_2': [1],
                  'dense_hidden_layers': [1],
                  'dense_layer_dim': [32],
                  'latent_dim': [16],
                  'batch_size': [20],
                  'epochs': [1000],
                  'patience': [30],
                  'optimizer': ['adam'],
                  'conv_activation': ['relu'],
                  'dense_activation': ['elu'],
                  'lr': [0.001],
                  'decay': [0.001],
                  'conv_kernel_init': ['he_normal'],
                  'dense_kernel_init': ['he_normal']
                  }

model = convolutional_vae.ConvolutionalVAE()
model.name = experiment_name
t = talos.Scan(x=ds.X_train,
               y=ds.X_train,
               model=model.training,
               experiment_name="vae_param_experiment",
               params=parameter_list,
               round_limit=1,
               print_params=True)

filehandler = open("talos_results/" + ds.name + ".obj", 'wb')
pickle.dump(t.data, filehandler)
