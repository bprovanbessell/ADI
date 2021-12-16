from models import convolutional_vae
from configurations import ds_config
from util import dataset
import os
import talos
import pickle

data_path = '/Users/andreavisentin/ADI/data_tm/'
data_path = '/home/avisentin/data/'
# 3 GPUS available on the cluster
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
experiment_name = 'All_measurements_sept_oct'

# Data creation and load
# Make new dataset configuration for all of these tests
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, experiment_name)
# ds = ds.data_load(ds.name)
ds.dataset_creation_influx()
ds.data_summary()

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

model = convolutional_vae.ConvolutionalVAE()
model.name = experiment_name
t = talos.Scan(x=ds.X_train,
               y=ds.X_train,
               model=model.training,
               experiment_name="vae_param_experiment",
               params=model.parameter_list,
               round_limit=100,
               print_params=True)

if not os.path.exists("talos_results/"):
    os.mkdir("talos_results")

filehandler = open("talos_results/" + ds.name + ".obj", 'wb')
pickle.dump(t.data, filehandler)