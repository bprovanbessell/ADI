from models import convolutional_vae
from configurations import ds_config
from util import dataset
import best_model_params
import os
import talos
import pickle
import time

data_path = '/Users/andreavisentin/ADI/data_tm/'
# 3 GPUS available on the cluster, we should use number 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# experiment_name = 'All_measurements_sept_oct_gcl_error'

# we want to train models for vibration, current, flux, and then all channels
experiment_name = "all_july_test_sep_nov_PU7001"
# experiment_name = "all_oct_18_gcl_error"
experiment_name = "flux_vib_grundfoss_9_feb"

# Data creation and load
# Make new dataset configuration for all of these tests
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, experiment_name)
# ds = ds.data_load(ds.name)
ds.dataset_creation_influx()
ds.data_summary()

# model_path = "saved_models/"
# model_path = "flux_final_model/"
# model_path = "vib_final_model/"
# model_path = "curr_final_model/"
# model_path = "all_model_params/"
# model_path = "all_model_PU7001/"
model_path = "flux_vib_model"

if not os.path.exists(model_path):
    os.mkdir(model_path)

t0 = time.time()

model = convolutional_vae.ConvolutionalVAE(model_path=model_path)
model.name = experiment_name

# when training the best model (with only one set of possible parameters),
# remember to set the round limit to 1, r remove it, otherwise it will throw an error
t = talos.Scan(x=ds.X_train,
               y=ds.X_train,
               model=model.training,
               experiment_name="vae_param_experiment",
               params=model.parameter_list,
               # params=best_model_params.all,
               round_limit=100,
               print_params=True)

if not os.path.exists("talos_results/"):
    os.mkdir("talos_results")

filehandler = open("talos_results/" + ds.name + ".obj", 'wb')
pickle.dump(t.data, filehandler)

t1 = time.time()

print("Total training time is: ", t1 - t0)
