from util import dataset
import time


class DatasetConfiguration:
    def __init__(self):
        self.local_token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
        self.gpu_token = "sPxOPI2tfrYVVjpC3b8IwxtJnv8ISRmTr_rEDaX4Q6WDj_SA1TjXPpplR26oJwHFB9aIei07jhsqHXdXkT6VnQ=="
        self.local_url = "http://localhost:8086"
        self.remote_url = "http://localhost:9090"

        # train test times for dataset common to all training and testing
        self.standard_time_train_start = time.mktime(time.strptime("10.06.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        self.standard_time_train_end = time.mktime(time.strptime("10.07.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        self.standard_time_test_start = time.mktime(time.strptime("10.07.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        self.standard_time_test_end = time.mktime(time.strptime("01.08.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

        return

    def SetConfiguration(self, ds, data_path, config_name='Vib_Grundfoss'):
        if config_name == 'Vib_Grundfoss':
            ds.name = 'Vib_Grundfoss'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.01.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}
        if config_name == 'Vib_Grundfoss_influx':
            ds.name = 'Vib_Grundfoss'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("27.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("27.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.05.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.nr_sample = 15000
            url = "http://localhost:8086"
            token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
            org = "Insight"
            bucket = "ADI"
            ds.read_write_dict = {"url": url,
                                  "token": token,
                                  "org": org,
                                  "bucket": bucket}
        if config_name == 'Flux_Grundfoss':
            ds.name = 'Flux_Grundfoss'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.01.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'Curr_Grundfoss':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset_ver.npy'
            ds.metadata_file = data_path + 'np_metadata_ver.npy'
            ds.signal = 'current'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("01.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.05.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}
        if config_name == 'EXa_1_Curr':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset_ver.npy'
            ds.metadata_file = data_path + 'np_metadata_ver.npy'
            ds.signal = 'current'
            ds.machine = 1
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("15.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("10.06.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://localhost:8086"
            token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
            org = "Insight"
            bucket = "ADI"
            ds.read_write_dict = {"url": url,
                                  "token": token,
                                  "org": org,
                                  "bucket": bucket}
        if config_name == 'EXa_1_Curr_small':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset_ver.npy'
            ds.metadata_file = data_path + 'np_metadata_ver.npy'
            ds.signal = 'current'
            ds.machine = 1
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("15.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("30.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://localhost:8086"
            token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
            org = "Insight"
            bucket = "ADI"
            ds.read_write_dict = {"url": url,
                                  "token": token,
                                  "org": org,
                                  "bucket": bucket}
        if config_name == 'EXa_1_Vib':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset_ver.npy'
            ds.metadata_file = data_path + 'np_metadata_ver.npy'
            ds.signal = 'vibration'
            ds.machine = 1
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("15.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("10.06.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        if config_name == 'EXa_1_Flux':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux'
            ds.machine = 1
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("15.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.08.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        if config_name == 'EXa_1_Flux_small':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux'
            ds.machine = 1
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("15.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("30.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://localhost:8086"
            token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
            org = "Insight"
            bucket = "ADI"
            ds.read_write_dict = {"url": url,
                                  "token": token,
                                  "org": org,
                                  "bucket": bucket}
        if config_name == 'Vib_Grundfoss_June':
            ds.name = 'Vib_Grundfoss'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("02.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("02.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.08.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        if config_name == "Test_tm_1":
            ds.name = 'Testing_testing'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.verbose = 1
            ds.time_train_start = time.mktime(time.strptime("01.08.2021 01:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.08.2021 02:20:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("01.08.2021 02:20:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.08.2021 02:40:00", "%d.%m.%Y %H:%M:%S"))
            ds.nr_sample = 15000
            url = "http://localhost:8086"
            token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
            org = "Insight"
            bucket = "test_bucket"
            ds.read_write_dict = {"url": url,
                                  "token": token,
                                  "org": org,
                                  "bucket": bucket}
            ds.channels = 2
        if config_name == "Test_ver_0":
            ds.name = 'Testing_testing'
            ds.data_file = data_path + 'np_dataset_ver.npy'
            ds.metadata_file = data_path + 'np_metadata_ver.npy'
            ds.signal = 'current'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.verbose = 1
            ds.time_train_start = time.mktime(time.strptime("01.08.2021 01:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.08.2021 02:20:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("01.08.2021 01:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.08.2021 01:20:00", "%d.%m.%Y %H:%M:%S"))
            ds.nr_sample = 15000
            url = "http://localhost:8086"
            token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
            org = "Insight"
            bucket = "test_bucket"
            ds.read_write_dict = {"url": url,
                                  "token": token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'All_measurements_test':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'all'
            ds.machine = 1
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("08.06.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("10.06.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("11.06.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("11.06.2021 12:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://localhost:9090"
            org = "Insight"
            bucket = "ADI"
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'All_measurements':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'all'
            ds.machine = 1
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = self.standard_time_train_start
            ds.time_train_end = self.standard_time_train_end
            ds.time_test_start = self.standard_time_test_start
            ds.time_test_end = self.standard_time_test_end

            ds.nr_sample = 15000
            url = "http://localhost:9090"
            org = "Insight"
            bucket = "ADI"
            ds.read_write_dict = {"url": self.local_url,
                                  "token": self.local_token,
                                  "org": org,
                                  "bucket": bucket}
        if config_name == 'All_measurements_sept_oct':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'all'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("31.08.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("30.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("30.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("31.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://localhost:9093"
            org = "Insight"
            bucket = "ADI"
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'All_measurements_nov_test':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'all'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("29.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("30.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("10.11.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("30.11.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://localhost:9093"
            org = "Insight"
            bucket = "ADI"
            ds.read_write_dict = {"url": self.remote_url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        # just test this dataset on the models we have trained for GCL already, we don't need new models
        # training for these models was actually september
        if config_name == 'curr_gcl_nov_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'current'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            # ds.time_train_start = time.mktime(time.strptime("15.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # ds.time_train_end = time.mktime(time.strptime("15.11.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # dummy values for now, using the pre trained models on the month of september
            ds.time_train_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # error occurred around 12 o' clock, it was a complete shutdown, So it should be visible in all of the measurements
            ds.time_test_start = time.mktime(time.strptime("10.11.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("10.12.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'vib_gcl_nov_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.10.2021 01:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("10.11.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("10.12.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'flux_gcl_nov_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.10.2021 01:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("10.11.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("10.12.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'all_gcl_nov_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'all'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.10.2021 01:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("10.11.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("10.12.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}


        # datasets for oct 18th error, train on just the month before

        if config_name == 'All_measurements_oct_18_gcl_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'all'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            # what training period here??
            # 1 month is probably enough
            ds.time_train_start = time.mktime(time.strptime("16.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("16.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # test just on 17th, 18th, 19th october
            ds.time_test_start = time.mktime(time.strptime("17.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("19.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'flux_oct_18_gcl_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            # Month of September to train
            ds.time_train_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # Anomaly occurred on 18/10, around 09:40
            ds.time_test_start = time.mktime(time.strptime("01.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("30.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'vib_oct_18_gcl_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            # what training period here??
            # 1 month is probably enough
            ds.time_train_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # test just on 17th, 18th, 19th october
            ds.time_test_start = time.mktime(time.strptime("01.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("30.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'curr_oct_18_gcl_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'current'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            # what training period here??
            # 1 month is probably enough
            ds.time_train_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # test just on 17th, 18th, 19th october
            ds.time_test_start = time.mktime(time.strptime("01.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("30.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'all_oct_18_gcl_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'all'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            # Month of September to train
            ds.time_train_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # Anomaly occurred on 18/10, around 09:40
            ds.time_test_start = time.mktime(time.strptime("01.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("30.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'flux_oct_18_gcl_error_zoom':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux'
            ds.machine = 2
            ds.normalization = 'scale'
            ds.speed_limit = 0
            # Month of September to train
            ds.time_train_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("02.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # Anomaly occurred on 18/10, around 09:40
            ds.time_test_start = time.mktime(time.strptime("18.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("23.10.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        """Train and test on a large period of normal behaviour, to see that the error does not change significantly"""
        if config_name == 'all_july_test_sep_nov_PU7001':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'all'
            ds.machine = 1
            ds.normalization = 'scale'
            ds.speed_limit = 0
            # Month of September to train
            ds.time_train_start = time.mktime(time.strptime("01.07.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("30.07.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # Anomaly occurred on 18/10, around 09:40
            ds.time_test_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.12.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}
        if config_name == 'all_july_test_sep_nov_grundfoss':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'all'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            # Month of September to train
            ds.time_train_start = time.mktime(time.strptime("01.07.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("30.07.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            # Anomaly occurred on 18/10, around 09:40
            ds.time_test_start = time.mktime(time.strptime("01.09.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.12.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'flux_vib_grundfoss_9_feb':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux_vib'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.01.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}

        if config_name == 'flux_vib_grundfoss_9_feb_zoom':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux_vib'
            ds.machine = 0
            ds.normalization = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.01.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("09.02.2021 10:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("09.02.2021 22:00:00", "%d.%m.%Y %H:%M:%S"))

            ds.nr_sample = 15000
            url = "http://143.239.81.3:8086"
            org = "Insight"
            bucket = "ADI"
            # train
            ds.read_write_dict = {"url": url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}
