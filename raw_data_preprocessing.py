from util import preprocessing
import numpy as np


# Path of the Otosense/TestMotors data
path_to_data = "/Volumes/Elements/ADI/data_tm20/"

# Machine names
devices = ["BlockAScrubber", "PU7001", "GeneralCoolingLoop"]


# Use this to create a numpy from all the json available in the folders
# Dataset contains the waveforms
# Metadata contains in order:
#                               index of the machine
#                               speed in RPM
#                               timestamp
dataset, metadata = preprocessing.npy_creation_otosense_new(path_to_data, devices)

# Dataset save
np.save(path_to_data +"np_dataset", dataset)
np.save(path_to_data +"np_metadata", metadata)


# Path of the Verdigris data
path_to_data = "/Volumes/Elements/ADI/data_verdigris/"

# Name of the motors (they are in the same order as the previous ones)
devices = ["JBE10001123", "JBE10001196", "JBE10001268"]
# The numpy arrays contain many columns, the one of interest start at these indexes
indexes = [6, 12, 9]

# Use this to create a numpy from all the .gz available in the folders
# Dataset contains the waveforms
# Metadata contains in order:
#                               index of the machine
#                               RPM not available in verdigris sensor
#                               timestamp
dataset, metadata = preprocessing.npy_creation_verdigris(path_to_data, devices, indexes)

# Dataset save
np.save(path_to_data +"np_dataset_ver", dataset)
np.save(path_to_data +"np_metadata_ver", metadata)
