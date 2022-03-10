"""
For sfpt verdigris server, we have to log in, get all the file info, and check if there is anything new yet

For otosense api, pull at a time to see if there is new files...

Ideally you wouldn't have to check, but just pull at a certain timeslot...

Schedule would be needed to pull at regular intervals.

"""

import time
from datetime import datetime, timedelta
import numpy as np

from models import convolutional_vae, pca


def get_last_interval_reconstruction_error():
    """

    """
    now = datetime.fromtimestamp(time.time())
    # should be rounded
    twenty_mins_ago = now - timedelta(minutes=20)
    floored = floor_minutes(twenty_mins_ago.minute)
    twenty_mins_ago = twenty_mins_ago.replace(minute=floored)

    # Do whatever you want with it, upload to influx, alert based on this error, etc...
    error = reconstruction_error_timed_interval_all(twenty_mins_ago)


def reconstruction_error_timed_interval_all(start_time: datetime):

    later = start_time + timedelta(minutes=20)
    # get the last otosense interval
    # get the last verdigris interval

    # reshape the intervals into size [1, 15000, 6]
    nr_sample = 15000
    # 6 channels for all
    channels = 6

    sample = np.zeros((1, nr_sample, channels))

    # Then run it on an appropriate VAE, and/or PCA model to get the reconstruction error
    vae = convolutional_vae.ConvolutionalVAE()
    model_name = "All_measurements_sept_oct_gcl_error0112"
    vae.load_models(model_name)

    reconstruction_error = model_mse(vae, sample)

    return reconstruction_error


def model_mse(model, x):
    return np.mean(tf.keras.losses.mean_squared_error(x, model.predict(x)), axis=1)


def floor_minutes(mins):
    """
    With the idea being that the interval is updated every twenty minutes exactly... This will have to be tested,
    as in practice the data may not be on time... Or there at all for that matter.
    """
    if mins < 20: return 0
    elif mins < 40: return 20
    else: return 40
