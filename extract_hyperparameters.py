from models import convolutional_vae, pca
from configurations import ds_config
from util import dataset, plotter
import time
import numpy as np

vae = convolutional_vae.ConvolutionalVAE(model_path="saved_models/")
model_name = "All_measurements_sept_oct_gcl_error0112"
vae.load_models(model_name)

print("model summary")
print(vae.model.summary())

print("encoder summary")
print(vae.encoder.summary())
print("decoder summary")
print(vae.decoder.summary())

print("model config")
print(vae.model.get_config())
print("encoder config")
print(vae.encoder.get_config())
print("decoder config")
print(vae.decoder.get_config())

print("model opt config")
print(vae.model.optimizer.get_config())
print("encoder opt config")
print(vae.encoder.optimizer.get_config())
print("decoder opt config")
print(vae.decoder.optimizer.get_config())

print("model loss")
print(vae.model.loss)
print("encoder loss")
print(vae.encoder.loss)
print("decoder loss")
print(vae.decoder.loss)