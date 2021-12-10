from models import convolutional_vae, pca, naive
from configurations import ds_config
from util import dataset, plotter
import numpy as np
import tensorflow as tf
import time

experiment_name = 'All_measurements'

# irrelevant when loading data from influx
data_path = '/Volumes/Elements/ADI/data_tm20/'

ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, 'All_measurements')

url = "http://localhost:8086"
token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
org = "Insight"
bucket = "ADI_results"
write_dict = {"url": url,
              "token": token,
              "org": org,
              "bucket": bucket}

# ds = ds.data_load(ds.name)
t0 = time.time()
ds.dataset_creation_influx()
t1 = time.time()

print("Minutes to load data: ", (t1 - t0) / 60)


pca = pca.PCAModel()
pca.training(ds.X_train, None, None, None, None)

p = plotter.Plotter()
p.name = ds.name
p.model = pca
p.X_train = np.asarray(ds.X_train)
p.X_test = np.asarray(ds.X_test)
p.meta_train = ds.metadata_train
p.meta_test = ds.metadata_test

p.reconstruction_error_time()
