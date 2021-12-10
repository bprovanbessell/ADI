from models import convolutional_vae
from configurations import ds_config
from util import dataset
import os
# import talos
import pickle

data_path = '/Users/andreavisentin/ADI/data_tm/'
data_path = './saved_data/'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
experiment_name = 'EXa_1_Curr'

# Data creation and load
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, experiment_name)
print(ds.name)
ds.data_load(ds.name)
ds.data_summary()

# model = convolutional_vae.ConvolutionalVAE()
# model.name = experiment_name
# t = talos.Scan(x=ds.X_test,
#                y=ds.X_test,
#                model=model.training,
#                experiment_name=experiment_name,
#                params=model.parameter_list,
#                round_limit=300,
#                print_params=True)
#
# filehandler = open("./talos_results/" + ds.name + ".obj", 'wb')
# pickle.dump(t.data, filehandler)
