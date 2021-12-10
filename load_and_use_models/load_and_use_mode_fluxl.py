from models import convolutional_vae, pca
from configurations import ds_config
from util import dataset, plotter
import time
import numpy as np

data_path = '/Volumes/Elements/ADI/data_tm20/'

# Data creation and load
ds = dataset.Dataset()
ds_config.DatasetConfiguration().SetConfiguration(ds, data_path, 'EXa_1_Flux_small')
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
# should be flux model
model_name = "Flux_Grundfoss0057"
vae.load_models(model_name)
# vae.load_models("Vib_Grundfoss0114")
# vae.load_models("EXa_1_Curr0127")


""" This part allows to plot an analysis without an anomaly
"""

p = plotter.Plotter()
p.name = ds.name
p.model = vae
p.X_train = np.asarray(ds.X_train)
p.X_test = np.asarray(ds.X_test)
p.meta_train = ds.metadata_train
p.meta_test = ds.metadata_test

p.reconstruction_error_time_influx(True, True, model_name, write_dict)

# # Plot the latent space
# p.rpm_time(train=False)
# p.latent_space_complete()
# p.plot_tsne(anomaly=False, train=False)
# p.reconstruction_error_time(test=False)
# p.reconstruction_error_time()
# p.roc()
# p.reconstruction_error(np.linspace(0, 3, 50))
# p.pdf(np.linspace(-16, 0, 50))

'''
""" This part allows to plot an analysis with an anomaly time
"""
err_time = time.mktime(time.strptime("09.02.2021 12:20:00", "%d.%m.%Y %H:%M:%S"))
rows = np.where(ds.metadata_test[:,2] <= err_time)
data = ds.X_test[rows,:,:]
data = data.reshape(data.shape[1:])
meta_test = ds.metadata_test[rows,:]
meta_test = meta_test.reshape(meta_test.shape[1:])
rows = np.where(ds.metadata_test[:,2] > err_time)
data_anomaly = ds.X_test[rows,:,:]
data_anomaly = data_anomaly.reshape(data_anomaly.shape[1:])
meta_anomaly = ds.metadata_test[rows,:]
meta_anomaly = meta_anomaly.reshape(meta_anomaly.shape[1:])


p = plotter.Plotter()
p.name = ds.name
p.model = vae
p.X_train = ds.X_train
p.X_test = data
p.X_anomaly = data_anomaly
p.meta_train = ds.metadata_train
p.meta_test = meta_test
p.meta_anomaly = meta_anomaly


# Plot the latent space
p.rpm_time()
p.latent_space_complete()
p.plot_tsne(anomaly=True, train=False)
p.plot_tsne(anomaly=True, train=True)
p.reconstruction_error_time()
p.reconstruction_error_time(train=False)
p.reconstruction_error_time(limit=1.5)
p.roc()
p.reconstruction_error(np.linspace(0, 3, 50))
p.pdf(np.linspace(-16, 0, 50))
'''

# begin_time = time.mktime(time.strptime("09.02.2021 11:00:00", "%d.%m.%Y %H:%M:%S"))
# rows = np.where(meta_test[:,2] >= begin_time)
# data = data[rows,:,:]
# data = data.reshape(data.shape[1:])
# meta_test = meta_test[rows,:]
# meta_test = meta_test.reshape(meta_test.shape[1:])
#
# end_time = time.mktime(time.strptime("09.02.2021 22:00:00", "%d.%m.%Y %H:%M:%S"))
# rows = np.where(meta_anomaly[:,2] < end_time)
# data_anomaly = data_anomaly[rows,:,:]
# data_anomaly = data_anomaly.reshape(data_anomaly.shape[1:])
# meta_anomaly = meta_anomaly[rows,:]
# meta_anomaly = meta_anomaly.reshape(meta_anomaly.shape[1:])
#
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
# p.reconstruction_error_time(train=False)
