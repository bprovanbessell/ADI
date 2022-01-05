from models import convolutional_vae, pca
from configurations import ds_config
from util import dataset, plotter
import time
import numpy as np

data_path = '/Volumes/Elements/ADI/data_tm20/'

# Data creation and load
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, 'All_measurements_sept_oct_gcl_error')

# ds_config.DatasetConfiguration().SetConfiguration(ds, data_path,'EXa_1_Curr')
#
# ds.dataset_creation()
# ds.data_save(ds.name)

gpu_token = "sPxOPI2tfrYVVjpC3b8IwxtJnv8ISRmTr_rEDaX4Q6WDj_SA1TjXPpplR26oJwHFB9aIei07jhsqHXdXkT6VnQ=="

url = "http://localhost:9093"
token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
org = "Insight"
bucket = "ADI_results"
write_dict = {"url": url,
              "token": gpu_token,
              "org": org,
              "bucket": bucket}

# ds = ds.data_load(ds.name)
t0 = time.time()
ds.dataset_creation_influx()
t1 = time.time()

print("Minutes to load data: ", (t1 - t0) / 60)

ds.data_summary()

vae = convolutional_vae.ConvolutionalVAE()
model_name = "All_measurements_sept_oct_gcl_error0112"
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

print("create graphs")
# # Plot the latent space
# p.rpm_time(train=False)
# p.latent_space_complete()
# p.plot_tsne(anomaly=False, train=False)
p.reconstruction_error_time(test=False)
p.reconstruction_error_time()
# p.roc()
# p.reconstruction_error(np.linspace(0, 3, 50))
# p.pdf(np.linspace(-16, 0, 50))

print("Model summary")
print(vae.model.summary)

print("upload reconstruction error to influx")
p.reconstruction_error_time_influx(True, True, model_name, write_dict)

print("train pca model")
pca = pca.PCAModel()
pca.training(ds.X_train, None, None, None, None)

model_name = "All_measurements_sept_oct_pca_gcl"


p = plotter.Plotter()
p.name = ds.name
p.model = pca
p.X_train = np.asarray(ds.X_train)
p.X_test = np.asarray(ds.X_test)
p.meta_train = ds.metadata_train
p.meta_test = ds.metadata_test

p.reconstruction_error_time()

p.reconstruction_error_time_influx(True, True, model_name, write_dict)

