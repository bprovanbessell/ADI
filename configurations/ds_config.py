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

        if config_name == 'vib_sept_oct_gcl_error':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 2
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
            # train
            ds.read_write_dict = {"url": self.remote_url,
                                  "token": self.gpu_token,
                                  "org": org,
                                  "bucket": bucket}
            # local test
            # ds.read_write_dict = {"url": url,
            #                       "token": self.gpu_token,
            #                       "org": org,
            #                       "bucket": bucket}

