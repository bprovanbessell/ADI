from util import preprocessing
import numpy as np

path_to_data = "/Users/andreavisentin/ADI/data_tm/"

path_to_data = "/Users/andreavisentin/Downloads/data-30-06-2021/"
path_to_data = "/Volumes/Elements/ADI/data_tm20/"
new_devices = ["BlockAScrubber", "PU7001", "GeneralCoolingLoop"]
old_devices = ["Grundfoss", "PU-7001", "GeneralCoolingLoop"]
file_types = ["completeSample", "operations", "performance"]
#
# # Use this to move Otosense data from the folder to the preprocessed one
# preprocessing.move_new_files(path_to_data, path_to_new_data, old_devices, new_devices, file_types)
#
# dataset, metadata = preprocessing.npy_creation_otosense_new(path_to_data, new_devices)
# preprocessing.plot_rpms(metadata)
# #
# # # Dataset save
# np.save(path_to_data +"np_dataset", dataset)
# np.save(path_to_data +"np_metadata", metadata)

path_to_data = "/Users/andreavisentin/ADI/data_verdigris/"

path_to_data = "/Volumes/Elements/ADI/data_verdigris/"
devices = ["JBE10001123", "JBE10001196", "JBE10001268"]
indexes = [6, 12, 9]

dataset, metadata = preprocessing.npy_creation_verdigris(path_to_data, devices, indexes)


np.save(path_to_data +"np_dataset_ver", dataset)
np.save(path_to_data +"np_metadata_ver", metadata)
