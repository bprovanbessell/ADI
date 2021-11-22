from util import dataset
import time


class DatasetConfiguration:
    def __init__(self):
        return

    def SetConfiguration(self, ds, data_path, config_name='Vib_Grundfoss'):
        if config_name == 'Vib_Grundfoss':
            ds.name = 'Vib_Grundfoss'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 0
            ds.normalizaion = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("01.01.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("01.02.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        if config_name == 'Flux_Grundfoss':
            ds.name = 'Flux_Grundfoss'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'flux'
            ds.machine = 0
            ds.normalizaion = 'scale'
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
            ds.normalizaion = 'scale'
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
            ds.normalizaion = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("15.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("10.06.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        if config_name == 'EXa_1_Vib':
            ds.name = config_name
            ds.data_file = data_path + 'np_dataset_ver.npy'
            ds.metadata_file = data_path + 'np_metadata_ver.npy'
            ds.signal = 'vibration'
            ds.machine = 1
            ds.normalizaion = 'scale'
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
            ds.normalizaion = 'scale'
            ds.speed_limit = 0
            ds.time_train_start = time.mktime(time.strptime("15.03.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_train_end = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_start = time.mktime(time.strptime("15.04.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
            ds.time_test_end = time.mktime(time.strptime("01.08.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
        if config_name == 'Vib_Grundfoss_June':
            ds.name = 'Vib_Grundfoss'
            ds.data_file = data_path + 'np_dataset.npy'
            ds.metadata_file = data_path + 'np_metadata.npy'
            ds.signal = 'vibration'
            ds.machine = 0
            ds.normalizaion = 'scale'
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
            ds.normalizaion = 'scale'
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
            ds.normalizaion = 'scale'
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
