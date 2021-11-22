import unittest
from TS.configurations import ds_config
from TS.util import dataset
from TS.influx.query_to_dataset import query_to_dataset
import numpy as np
import time
from sklearn.preprocessing import StandardScaler


class QueryTest(unittest.TestCase):
    """
    Basically check that the train set is created correctly by the querying
    Check the result of creating it with the dataset create function, and with the query function

    Will have to check before and after the scalar normalization as well
    """

    def test_tm_0(self):
        data_path = "../data/test-motors/data-31-08-2021/"

        test_dataset_baseline = dataset.Dataset()
        ds_config.DatasetConfiguration().SetConfiguration(test_dataset_baseline, data_path, 'Test_tm_1')

        test_dataset_query = dataset.Dataset()
        ds_config.DatasetConfiguration().SetConfiguration(test_dataset_query, data_path, 'Test_tm_1')
        # make the datasets
        # get the intermediate steps here
        # test_dataset_baseline.dataset_creation()
        baseline_x_train = dataset_intermediate(test_dataset_baseline)

        # it is one hour later, maybe to do with daylight savings
        # (not problem with writing script, which has been tested)
        test_dataset_query.time_train_start = time.mktime(time.strptime("01.08.2021 02:00:00", "%d.%m.%Y %H:%M:%S"))
        test_dataset_query.time_train_end = time.mktime(time.strptime("01.08.2021 03:20:00", "%d.%m.%Y %H:%M:%S"))

        query_x_train = query_to_dataset(test_dataset_query.time_train_start,
                                         test_dataset_query.time_train_end,
                                         test_dataset_query.signal,
                                         test_dataset_query.nr_sample,
                                         test_dataset_query.machine,
                                         test_dataset_query.speed_limit,
                                         test_dataset_query.read_write_dict)

        self.assertEqual(baseline_x_train.shape, query_x_train.shape)
        np.testing.assert_allclose(baseline_x_train, query_x_train)

        test_dataset_query.X_train = query_x_train
        test_dataset_query.channels = 2
        norm(test_dataset_query)
        norm(test_dataset_baseline)

        np.testing.assert_allclose(test_dataset_baseline.X_train, test_dataset_query.X_train)

    def test_tm_2(self):
        data_path = "../data/test-motors/data-31-08-2021/"

        test_dataset_baseline = dataset.Dataset()
        ds_config.DatasetConfiguration().SetConfiguration(test_dataset_baseline, data_path, 'Test_tm_1')

        test_dataset_query = dataset.Dataset()
        ds_config.DatasetConfiguration().SetConfiguration(test_dataset_query, data_path, 'Test_tm_1')
        # make the datasets
        # it is one hour later, maybe to do with daylight savings
        # (not problem with writing script, which has been tested)
        test_dataset_query.time_train_start = time.mktime(time.strptime("01.08.2021 02:00:00", "%d.%m.%Y %H:%M:%S"))
        test_dataset_query.time_train_end = time.mktime(time.strptime("01.08.2021 03:20:00", "%d.%m.%Y %H:%M:%S"))

        test_dataset_query.time_test_start = time.mktime(time.strptime("01.08.2021 03:20:00", "%d.%m.%Y %H:%M:%S"))
        test_dataset_query.time_test_end = time.mktime(time.strptime("01.08.2021 03:40:00", "%d.%m.%Y %H:%M:%S"))

        test_dataset_baseline.dataset_creation()
        test_dataset_query.dataset_creation_influx()

        self.assertEqual(test_dataset_baseline.X_train.shape, test_dataset_query.X_train.shape)
        np.testing.assert_allclose(test_dataset_baseline.X_train, test_dataset_query.X_train)

    def test_verdigris(self):
        # Similar to above, but checks verdigris data
        data_path = "../data/verdigris/"

        test_dataset_baseline = dataset.Dataset()
        ds_config.DatasetConfiguration().SetConfiguration(test_dataset_baseline, data_path, 'Test_ver_0')

        test_dataset_query = dataset.Dataset()
        ds_config.DatasetConfiguration().SetConfiguration(test_dataset_query, data_path, 'Test_ver_0')

        # add the extra hour, still not entirely sure why this has to happen
        # Could be some weird issue with UTC and daylight savings
        test_dataset_query.time_train_start = time.mktime(time.strptime("01.08.2021 02:00:00", "%d.%m.%Y %H:%M:%S"))
        test_dataset_query.time_train_end = time.mktime(time.strptime("01.08.2021 03:20:00", "%d.%m.%Y %H:%M:%S"))

        test_dataset_query.time_test_start = time.mktime(time.strptime("01.08.2021 02:00:00", "%d.%m.%Y %H:%M:%S"))
        test_dataset_query.time_test_end = time.mktime(time.strptime("01.08.2021 02:20:00", "%d.%m.%Y %H:%M:%S"))

        test_dataset_baseline.dataset_creation()
        test_dataset_query.dataset_creation_influx()

        print(test_dataset_baseline.X_train)
        print(test_dataset_query.X_train)

        self.assertEqual(test_dataset_baseline.X_train.shape, test_dataset_query.X_train.shape)
        np.testing.assert_allclose(test_dataset_baseline.X_train, test_dataset_query.X_train)


def dataset_intermediate(ds):
    X = np.load(ds.data_file)
    print(X.shape)
    metadata = np.load(ds.metadata_file)
    if ds.speed_limit > 0:
        active = metadata[:, 1] > ds.speed_limit
        X = X[active, :, ]
        metadata = metadata[active, :]
    if ds.signal != "current":
        X = np.moveaxis(X, 1, 2)
    print(X.shape)

    if ds.verbose:
        print("Selection of the signal and machine")
    if ds.signal == "flux":
        X = X[:, :, 0]
        X = X.reshape((X.shape[0], X.shape[1], 1))

    if ds.signal == "vibration":
        X = X[:, :, 1:]
    ds.channels = X.shape[-1]

    active = metadata[:, 0] == ds.machine
    X = X[active, :, ]
    metadata = metadata[active, :]

    if ds.verbose:
        print("Data size ", X.shape)

    if ds.verbose:
        print("Selection of the timeframe")
        print(ds.time_train_start)
        print(ds.time_train_end)
    # Select the right timeframe for the training set
    active = np.where(np.logical_and(metadata[:, 2] > ds.time_train_start, metadata[:, 2] < ds.time_train_end))
    ds.X_train = X[active, :, ]
    ds.X_train = ds.X_train.reshape(ds.X_train.shape[1:])
    ds.metadata_train = metadata[active, :]
    ds.metadata_train = ds.metadata_train.reshape(ds.metadata_train.shape[1:])

    if ds.verbose:
        print("Train size ", ds.X_train.shape)

    return ds.X_train


def norm(ds):
    if ds.verbose:
        print("Data normalization")
    # Normalization
    ds.scalers = {}
    for i in range(ds.channels):
        ds.scalers[i] = StandardScaler()
        ds.X_train[:, :, i] = ds.scalers[i].fit_transform(ds.X_train[:, :, i])
    # for i in range(ds.channels):
    #     ds.X_test[:, :, i] = ds.scalers[i].transform(ds.X_test[:, :, i])
