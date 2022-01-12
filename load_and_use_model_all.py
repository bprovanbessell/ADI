from models import convolutional_vae, pca
from configurations import ds_config
from util import dataset, plotter
import time
import numpy as np

data_path = '/Volumes/Elements/ADI/data_tm20/'

# Data creation and load
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, 'All_measurements_sept_oct_test')

# ds_config.DatasetConfiguration().SetConfiguration(ds, data_path,'EXa_1_Curr')
#
# ds.dataset_creation()
# ds.data_save(ds.name)

gpu_token = "sPxOPI2tfrYVVjpC3b8IwxtJnv8ISRmTr_rEDaX4Q6WDj_SA1TjXPpplR26oJwHFB9aIei07jhsqHXdXkT6VnQ=="

url = "http://localhost:9090"
token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
org = "Insight"
bucket = "ADI_results"
write_dict = {"url": url,
              "token": gpu_token,
              "org": org,
              "bucket": bucket}

# ds = ds.data_load(ds.name)
t0 = time.time()
# ds.dataset_creation_influx()
t1 = time.time()

print("Minutes to load data: ", (t1 - t0) / 60)

# ds.data_summary()

vae = convolutional_vae.ConvolutionalVAE()
# model_name = "All_measurements_sept_oct_gcl_error0112"
# vae.load_models(model_name)

# get the config
vae.load_models("EXa_1_Curr0127")
print("model summary")
print(vae.model.summary())

print("encoder summary")
print(vae.encoder.summary())

print(vae.decoder.summary())


""" This part allows to plot an analysis without an anomaly
"""

# p = plotter.Plotter()
# p.name = ds.name
# p.model = vae
# p.X_train = np.asarray(ds.X_train)
# p.X_test = np.asarray(ds.X_test)
# p.meta_train = ds.metadata_train
# p.meta_test = ds.metadata_test
#
# print("create graphs")
# # Plot the latent space
# # p.rpm_time(train=False)
# # p.latent_space_complete()
# # p.plot_tsne(anomaly=False, train=False)
# p.reconstruction_error_time(train=False)
# p.reconstruction_error(np.linspace(0, 3, 50))
# p.reconstruction_error_time()
# p.reconstruction_error_time_moving_avg()
# p.roc()
# p.reconstruction_error(np.linspace(0, 3, 50))
# p.pdf(np.linspace(-16, 0, 50))

"""Plotting with an anomaly"""

# err_time_start = time.mktime(time.strptime("18.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
# # err_time_end = time.mktime(time.strptime("20.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
# # anomaly_rows = np.where(np.logical_and(ds.metadata_test[:, 2] >= err_time_start, ds.metadata_test[:, 2] <= err_time_end))
# rows = np.where(ds.metadata_test[:,2] <= err_time_start)
# data = ds.X_test[rows,:,:]
# data = data.reshape(data.shape[1:])
# meta_test = ds.metadata_test[rows,:]
# meta_test = meta_test.reshape(meta_test.shape[1:])
# rows = np.where(ds.metadata_test[:,2] > err_time_start)
# data_anomaly = ds.X_test[rows,:,:]
# data_anomaly = data_anomaly.reshape(data_anomaly.shape[1:])
# meta_anomaly = ds.metadata_test[rows,:]
# meta_anomaly = meta_anomaly.reshape(meta_anomaly.shape[1:])
#
# print("Model summary")
# print(vae.model.summary())
#
# p = plotter.Plotter()
# p.name = ds.name
# p.model = vae
# p.X_train = ds.X_train
# p.X_test = data
# p.X_anomaly = data_anomaly
# p.meta_train = ds.metadata_train
# p.meta_test = meta_test
# p.meta_anomaly = meta_anomaly


# Plot the latent space
# p.rpm_time()
# p.latent_space_complete()
# p.plot_tsne(anomaly=True, train=False)
# p.plot_tsne(anomaly=True, train=True)

# p.reconstruction_error_time(anomaly=True)

# p.reconstruction_error_time(train=False)
# p.reconstruction_error_time(limit=1.5)
# p.roc()

# p.reconstruction_error(np.linspace(0, 3, 50), anomaly=True)
# p.reconstruction_error_time_moving_avg(anomaly=True)

# p.pdf(np.linspace(-16, 0, 50))

# print("upload reconstruction error to influx")
# p.reconstruction_error_time_influx(True, True, model_name, write_dict)

# print("train pca model")
# pca = pca.PCAModel()
# pca.training(ds.X_train, None, None, None, None)
#
# model_name = "All_measurements_sept_oct_pca_gcl"
#
#
# p = plotter.Plotter()
# p.name = ds.name
# p.model = pca
# p.X_train = np.asarray(ds.X_train)
# p.X_test = np.asarray(ds.X_test)
# p.meta_train = ds.metadata_train
# p.meta_test = ds.metadata_test
#
# p.reconstruction_error_time()

# p.reconstruction_error_time_influx(True, True, model_name, write_dict)

