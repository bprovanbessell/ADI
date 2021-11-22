import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import random
import math


class Dataset:
    def __init__(self):
        # Definition of all the instance attributes
        # Name of the experiment
        self.name = "Dataset"
        # Training instances
        self.X_train = []
        # Test instances
        self.X_test = []
        # Training labels
        self.y_train = []
        # Test labels
        self.y_test = []

        # Training metadata
        self.metadata_train = []
        # Test metadata
        self.metadata_test = []

        # Input files
        self.data_file = None
        self.metadata_file = None
        self.data_path = './saved_data/'

        # Type of signal to process
        self.signal = 'flux'
        # Machine involved in the experiment
        self.machine = "Grundfoss"

        # Type of  data normalization used
        self.normalizaion = "scale"
        self.scalers = []

        # Speed filter, samples under this are removed
        self.speed_limit = -1

        # Training time start
        self.time_train_start = time.mktime(time.strptime("01.01.2020 00:00:00", "%d.%m.%Y %H:%M:%S"))
        # Training time end
        self.time_train_end = time.mktime(time.strptime("01.01.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

        # Test time start
        self.time_test_start = time.mktime(time.strptime("01.01.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        # Test time end
        self.time_test_end = time.mktime(time.strptime("01.01.3000 00:00:00", "%d.%m.%Y %H:%M:%S"))

        # Number of waveforms saved
        self.channels = 3
        # Number of samples for each waveform
        self.samp_freq = 6250
        self.samp_time = 2.4
        self.nr_sample = (int)(self.samp_freq * self.samp_time)

        self.verbose = 1

    def data_save(self, name):
        with open(self.data_path + name, 'wb') as file:
            # Step 3
            pickle.dump(self, file)
            print("File saved in " + self.data_path + name)

    def data_load(self, name):
        with open(self.data_path + name, 'rb') as file:
            # Step 3
            return pickle.load(file)

    def data_summary(self):
        # print(type(self.X_test))
        # print(self.X_test)
        print('Training', self.X_train.shape, 'Testing', self.X_test.shape)

    def dataset_creation(self):
        if self.data_file is None or self.metadata_file is None:
            print("ERROR: Source files not specified")
            return
        if self.verbose:
            print("Data load")
        X = np.load(self.data_file)
        print(X.shape)
        metadata = np.load(self.metadata_file)
        if self.speed_limit > 0:
            active = metadata[:, 1] > self.speed_limit
            X = X[active, :, ]
            metadata = metadata[active, :]
        if self.signal != "current":
            X = np.moveaxis(X, 1, 2)
        print(X.shape)

        if self.verbose:
            print("Selection of the signal and machine")
        if self.signal == "flux":
            X = X[:, :, 0]
            X = X.reshape((X.shape[0], X.shape[1], 1))

        if self.signal == "vibration":
            X = X[:, :, 1:]

        self.channels = X.shape[-1]

        active = metadata[:, 0] == self.machine
        X = X[active, :, ]
        metadata = metadata[active, :]


        if self.verbose:
            print("Data size ", X.shape)

        if self.verbose:
            print("Selection of the timeframe")
            print(self.time_train_start)
            print(self.time_train_end)
        # Select the right timeframe for the training set
        active = np.where(np.logical_and(metadata[:, 2] > self.time_train_start, metadata[:, 2] < self.time_train_end))
        self.X_train = X[active, :, ]
        self.X_train = self.X_train.reshape(self.X_train.shape[1:])
        self.metadata_train = metadata[active, :]
        self.metadata_train = self.metadata_train.reshape(self.metadata_train.shape[1:])

        if self.verbose:
            print("Train size ", self.X_train.shape)

        # Select the right timeframe for the test set
        active = np.where(np.logical_and(metadata[:, 2] > self.time_test_start, metadata[:, 2] < self.time_test_end))
        self.X_test = X[active, :, ]
        self.X_test = self.X_test.reshape(self.X_test.shape[1:])
        self.metadata_test = metadata[active, :]
        self.metadata_test = self.metadata_test.reshape(self.metadata_test.shape[1:])

        if self.verbose:
            print("Test size ", self.X_test.shape)

        if self.verbose:
            print("Data normalization")
        # Normalization
        self.scalers = {}
        for i in range(self.channels):
            self.scalers[i] = StandardScaler()
            self.X_train[:, :, i] = self.scalers[i].fit_transform(self.X_train[:, :, i])
        for i in range(self.channels):
            self.X_test[:, :, i] = self.scalers[i].transform(self.X_test[:, :, i])
