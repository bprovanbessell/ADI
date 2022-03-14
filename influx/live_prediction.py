"""
For sfpt verdigris server, we have to log in, get all the file info, and check if there is anything new yet

For otosense api, pull at a time to see if there is new files...

Ideally you wouldn't have to check, but just pull at a certain timeslot...

Schedule would be needed to pull at regular intervals.


Method to pull data from the single verdigris file
Error checking if the file does not exist
"""

import time
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from models import convolutional_vae, pca
import verdigris_api
from otosense_api import OtosenseApi


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


def reconstruction_error_timed_interval_all(start_time: datetime, device_id):
    # check both timestamps are the same, etc...
    # check that there is no nan or missing values in the datasets, as the error is nan
    later = start_time + timedelta(minutes=20)

    # get the last verdigris interval
    indexes = [6, 12, 9]

    # works fine
    # ver_fn = verdigris_api.get_file_from_time(device_id, start_time, temp_file=True)
    ver_fn = "/tmp/JBE10001268.2022-01-150000-XXXX.gz"
    ver_dataset = verdigris_api.get_verdigris_dataset(ver_fn, device_id, indexes)

    print("verdigris shape")
    print(ver_dataset.shape)
    # print(np.count_nonzero(np.isnan(ver_dataset)))

    # get the last otosense interval
    tm_devices_api = ["Block A Scrubber", "PU7001", "General Cooling Loop"]

    # seems fine
    api = OtosenseApi()
    oto_dataset = api.get_single_sample(start_time, device_id, motor_name=tm_devices_api[device_id])

    print("otosense shape")
    print(oto_dataset.shape)
    # print(np.count_nonzero(np.isnan(oto_dataset)))

    # reshape the intervals into size [1, 15000, 6]
    nr_sample = 15000
    # 6 channels for all
    channels = 6

    sample = np.zeros((1, nr_sample, channels))
    sample[0, :, :3] = oto_dataset[0]
    sample[0, :, 3:6] = ver_dataset[0]

    print("normalization")
    # Normalization
    scalers = {}
    for i in range(channels):
        scalers[i] = StandardScaler()
        sample[:, :, i] = scalers[i].fit_transform(sample[:, :, i])

    print("Sample shape")
    print(sample.shape)

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


if __name__ == "__main__":
    start = time.mktime(time.strptime("15.01.2022 00:00:00", "%d.%m.%Y %H:%M:%S"))
    # This one should be on for otosense
    # There should be data up to 17-01 for all verdigris
    device_id = 2

    start_dt = datetime.fromtimestamp(start)

    print("generating error")
    error = reconstruction_error_timed_interval_all(start_dt, device_id)

    print("error: ", error)
