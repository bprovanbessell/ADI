from models import convolutional_vae, pca, naive
from configurations import ds_config
from util import dataset, plotter
import numpy as np
import tensorflow as tf
import time

experiment_name = 'Vib_Grundfoss'
input_shape = (1, 15000, 2)
test_limit = 1443
data_path = '/Users/andreavisentin/ADI/data_tm/'

# Data creation and load
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, experiment_name)
ds = ds.data_load(ds.name)
ds.data_summary()

v = convolutional_vae.ConvolutionalVAE()
v.load_models("Vib_Grundfoss0246")

p = pca.PCAModel()
p.training(ds.X_train, None, None, None, None)

n = naive.NaiveCompression()
n.input_shape = input_shape
n.training(ds.X_train, None, None, None, None)

solvers = [v, p, n]
err_solvers = plotter.reconstruction_comparison(solvers, ds.X_test, test_limit, input_shape)

err_time = time.mktime(time.strptime("10.02.2021 09:40:00", "%d.%m.%Y %H:%M:%S"))
metadata_fault = ds.metadata_test[:,2] > err_time

err_solvers = plotter.reconstruction_comparison_anomaly(solvers, ds.X_test, test_limit, input_shape,metadata_fault)
