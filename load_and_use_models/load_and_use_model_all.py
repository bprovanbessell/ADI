from models import convolutional_vae, pca
from configurations import ds_config
from util import dataset, plotter
import time
import numpy as np

data_path = '/Volumes/Elements/ADI/data_tm20/'

# Data creation and load
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, 'All_measurements')
# ds_config.DatasetConfiguration().SetConfiguration(ds, data_path,'EXa_1_Curr')
#
# ds.dataset_creation()
# ds.data_save(ds.name)

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

ds.data_summary()

vae = convolutional_vae.ConvolutionalVAE()
model_name = "All_measurements0109"
vae.load_models(model_name)


""" This part allows to plot an analysis without an anomaly
"""

p = plotter.Plotter()
p.name = ds.name
p.model = vae
p.X_train = np.asarray(ds.X_train)
p.X_test = np.asarray(ds.X_test)
p.meta_train = ds.metadata_train
p.meta_test = ds.metadata_test

# p.reconstruction_error_time_influx(True, True, model_name, write_dict)

# # Plot the latent space
p.rpm_time(train=False)
p.latent_space_complete()
p.plot_tsne(anomaly=False, train=False)
p.reconstruction_error_time(test=False)
p.reconstruction_error_time()
p.roc()
p.reconstruction_error(np.linspace(0, 3, 50))
p.pdf(np.linspace(-16, 0, 50))