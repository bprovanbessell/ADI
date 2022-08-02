from models import convolutional_vae, pca
from configurations import ds_config
from util import dataset, plotter
import time
import numpy as np
import tensorflow as tf

data_path = '/Volumes/Elements/ADI/data_tm20/'

# Data creation and load
ds = dataset.Dataset()
# ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, 'vib_gcl_nov_error')

# experiment_name = "curr_nov_gcl_error"
# experiment_name = "all_oct_18_gcl_error"

# For 9th of february
experiment_name = "Vib_Grundfoss"
# and
experiment_name = "Flux_Grundfoss"

experiment_name = "all_july_test_sep_nov_PU7001"

experiment_name = "flux_vib_grundfoss_9_feb_zoom"

# experiment_name = "all_july_test_sep_nov_grundfoss"

ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, experiment_name)

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
ds.dataset_creation_influx()
t1 = time.time()

print("Minutes to load data: ", (t1 - t0) / 60)

ds.data_summary()

# regen the results for curr (In prog), vib, flux, and all model (When done training)

# model_path = "saved_models/"
# model_path = "curr_final_model/"
# model_path = "vib_final_model/"
# model_path = "flux_final_model/"
# model_path = "all_model_params/"

model_path = "9feb_models/"
model_path = "all_model_PU7001/"
model_path = "flux_vib_model/"
# model_path = "all_model_grundfoss/"
vae = convolutional_vae.ConvolutionalVAE(model_path=model_path)

# so for all the different models, vibration, flux and current
# we need the same results on each test set
# plots of reconstruction error on the test set
# compare it to PCA too

# we also need the layout/architecture of the model
# model_name = "All_measurements_sept_oct_gcl_error0112"
# model_name = "curr_oct_18_gcl_error0029"
# model_name = "vib_oct_18_gcl_error0012"
# model_name = 'flux_oct_18_gcl_error0042'
# model_name = 'all_oct_18_gcl_error0121'

model_name = "Vib_Grundfoss0114"
model_name = "Flux_Grundfoss0057"
model_name = "all_july_test_sep_nov_PU70010473"
model_name = "flux_vib_modelflux_vib_grundfoss_9_feb0226"
# model_name = "all_july_test_sep_nov_grundfoss0492"

vae.load_models(model_name)

# get the config
# vae.load_models("EXa_1_Curr0127")
print("model summary")
print(vae.model.summary())

print("encoder summary")
print(vae.encoder.summary())

print(vae.decoder.summary())

"""Plot without an anomaly"""
'''
p = plotter.Plotter()
p.name = "VAE - All Measurements"
p.model = vae
p.X_train = np.asarray(ds.X_train)
p.X_test = np.asarray(ds.X_test)
p.meta_train = ds.metadata_train
p.meta_test = ds.metadata_test

p.latent_space_complete(anomaly=False)
p.plot_tsne(anomaly=False, train=True, after_anomaly=False)
p.reconstruction_error_time(anomaly=False, train=True, after_anomaly=False)
p.reconstruction_error_time(anomaly=False, train=False, after_anomaly=False)

p.reconstruction_error(np.linspace(0, 3, 50), anomaly=False, train=True, after_anomaly=False)

# absolute vibration
p.mean_absolute_vibration(train=True, test=True, anomaly=False)
p.mean_absolute_vibration(train=False, test=True)
'''
"""Plotting with an anomaly"""


# 18th october error
err_time_start = time.mktime(time.strptime("18.10.2021 09:20:00", "%d.%m.%Y %H:%M:%S"))
err_time_end = time.mktime(time.strptime("20.10.2021 21:00:00", "%d.%m.%Y %H:%M:%S"))

# 25th november error
# err_time_start = time.mktime(time.strptime("25.11.2021 13:40:00", "%d.%m.%Y %H:%M:%S"))
# might need to change this, still not sure exactly when this ended,
# we may want to just remove the after anomaly segment,as it my not go back to normal for the data we have
# err_time_end = time.mktime(time.strptime("30.11.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

# 9th february error, ADI says its at 13:25, but reconstruction error can pinpoint it to 12:20
# So we want to reproduce the graphs

# 1, general reconstruction over tie both with and without training, reconstruction error, makes ure PCA is in there too
# 3, zoom in on the time, from 11 to 22 that day
# 4 compare tsne again

# Not really cleare when the error ended, as the motor followed a different pattern of operation.
err_time_start = time.mktime(time.strptime("09.02.2021 12:00:00", "%d.%m.%Y %H:%M:%S"))
err_time_end = time.mktime(time.strptime("09.02.2021 16:00:00", "%d.%m.%Y %H:%M:%S"))



# test set before anomaly
rows = np.where(ds.metadata_test[:,2] <= err_time_start)
data = ds.X_test[rows,:,:]
data = data.reshape(data.shape[1:])
meta_test = ds.metadata_test[rows,:]
meta_test = meta_test.reshape(meta_test.shape[1:])
# test set at anomaly
# could try this as well
# anomaly_rows = np.where(np.logical_and(ds.metadata_test[:, 2] >= err_time_start, ds.metadata_test[:, 2] <= err_time_end))

rows = np.where((ds.metadata_test[:,2] > err_time_start) & (ds.metadata_test[:, 2] <= err_time_end))
data_anomaly = ds.X_test[rows,:,:]
data_anomaly = data_anomaly.reshape(data_anomaly.shape[1:])
meta_anomaly = ds.metadata_test[rows,:]
meta_anomaly = meta_anomaly.reshape(meta_anomaly.shape[1:])
# test set after anomaly
rows = np.where(ds.metadata_test[:,2] > err_time_end)
data_after = ds.X_test[rows,:,:]
data_after = data_after.reshape(data_after.shape[1:])
meta_after = ds.metadata_test[rows,:]
meta_after = meta_after.reshape(meta_after.shape[1:])

# The data after the anomaly should be integrated into all the plotting functions

# print("Model summary")
# print(vae.model.summary())

p = plotter.Plotter()
# p.name = ds.name + "Vae"
p.name = "- VAE - Flux and Vibration"
p.model = vae
p.X_train = ds.X_train
p.X_test = data
p.X_anomaly = data_anomaly
p.X_after = data_after
p.meta_train = ds.metadata_train
p.meta_test = meta_test
p.meta_anomaly = meta_anomaly
p.meta_after = meta_after

# Plot the latent space
# p.rpm_time()
p.latent_space_complete(anomaly=True)
p.latent_space_complete(anomaly=False)
p.plot_tsne(anomaly=True, train=False, after_anomaly=True)
p.plot_tsne(anomaly=True, train=True, after_anomaly=True)

# p.reconstruction_error_time(anomaly=True)

# p.reconstruction_error_time(train=False)
p.reconstruction_error_time(anomaly=True, train=False, after_anomaly=True)
p.reconstruction_error_time(anomaly=True, train=True, after_anomaly=True)
p.reconstruction_error_time(limit=1.5)
# p.roc()
# for some reason only this is working
p.reconstruction_error(np.linspace(0, 3, 50), anomaly=True, train=True, after_anomaly=True)
p.reconstruction_error(np.linspace(0, 3, 50), anomaly=True, train=False, after_anomaly=True)

# Compute the time to infer a number of points

# tinfer0 = time.time()
# np.mean(tf.keras.losses.mean_squared_error(ds.X_test, vae.model.predict(ds.X_test)), axis=1)
#
# tinfer1 = time.time()
#
# print("time to infer test points: ", tinfer1 - tinfer0)
#
# print(ds.X_train.shape)
# print("time to infer a single point (avg)", (tinfer1 - tinfer0)/ds.X_train.shape[0])


# p.reconstruction_error_time_moving_avg(anomaly=True)

# p.pdf(np.linspace(-16, 0, 50))

# print("upload reconstruction error to influx")
# p.reconstruction_error_time_influx(True, True, model_name, write_dict)

# look into code of PCA to make sure it all runs smoothly
print("train pca model")
pca = pca.PCAModel()
pca.training(ds.X_train, None, None, None, None)

# p = plotter.Plotter()

# p.name = ds.name + "pca"
p.name = "- PCA - Flux and Vibration"
p.model = pca
# p.X_train = np.asarray(ds.X_train)
# p.X_test = np.asarray(ds.X_test)
# p.meta_train = ds.metadata_train
# p.meta_test = ds.metadata_test

# Add the same plots that we do for the vae models

p.reconstruction_error_time(anomaly=True, train=False, after_anomaly=True)
p.reconstruction_error_time(anomaly=True, train=True, after_anomaly=True)
# p.reconstruction_error_time(limit=1.5)
# p.roc()
# for some reason only this is working
# p.reconstruction_error(np.linspace(0, 3, 50), anomaly=True)
p.reconstruction_error(np.linspace(0, 3, 50), anomaly=True, train=True, after_anomaly=True)
p.reconstruction_error(np.linspace(0, 3, 50), anomaly=True, train=False, after_anomaly=True)
# p.reconstruction_error_time_moving_avg(anomaly=True)'''
