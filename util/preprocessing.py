# Read the file names and timestamps
import re
import glob
import json
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import gzip
import pandas as pd
import glob
# Data import

# Used for the old format of Otosense data. ignore it
def move_new_files(path_to_data, path_to_raw_data, old_devices, new_devices, file_types):
    for i, d in enumerate(new_devices):
        for ft in file_types:
            for file_path in glob.glob(path_to_raw_data + d + "/*" + ft + "*"):
                print(file_path)
                new_path = path_to_data + old_devices[i] + "/" + ft + "/" + re.split("/", file_path)[-1]
                print(new_path)
                os.rename(file_path, new_path)

# Used for the old format of Otosense data. ignore it
def npy_creation_otosense(path_to_data, devices):
    f_lib = []
    timestamp = []
    for device in devices:
        for filename in sorted(glob.glob(path_to_data + device + "/completeSample/data*")):
            try:
                f_lib.append(filename)
                timestamp.append(
                    datetime.datetime.fromtimestamp(int(re.split("[# .]", filename)[7]) / 1000)
                )
            except Exception as E:
                print("warning:" + filename)
    print("Total files found: ", len(f_lib))

    # Number of waveforms saved
    nr_channels = 3
    # Number of samples for each waveform
    nr_samples = 15000

    dataset = np.zeros((len(f_lib), nr_channels, nr_samples))
    metadata = np.zeros((len(f_lib), 3))

    for i in range(len(f_lib)):
        print("File " + str(i) + "of" + str(len(f_lib)), end="\r")

        with open(f_lib[i]) as f:
            data = json.load(f)
            dataset[i, 0, :] = data["flux"]
            dataset[i, 1, :] = data["vibx"]
            dataset[i, 2, :] = data["vibz"]
        with open(f_lib[i].replace("completeSample", "performance")) as f:
            data = json.load(f)
            for j, device in enumerate(devices):
                if device in f_lib[i]:
                    metadata[i, 0] = j
            metadata[i, 1] = data["rpm"]
            metadata[i, 2] = datetime.datetime.timestamp(timestamp[i])

    return dataset, metadata



def npy_creation_otosense_new(path_to_data, devices):
    f_lib = []
    timestamp = []
    for device in devices:
        path = path_to_data + device + "/data*"
        print(path)
        filelist = sorted(glob.glob(path))
        for filename in filelist:
            if "flux" in filename:
                try:
                    if filename.replace("flux", "vibx") in filelist and filename.replace("flux", "vibz") in filelist:
                        f_lib.append(filename)
                        time_text = int(re.split("[# .]", filename)[-2])
                        timestamp.append(
                            datetime.datetime.fromtimestamp(time_text // 1000)
                        )
                    else:
                        print("Missing vib:" + filename )
                except Exception as E:
                    print("warning:" + filename + "\nTime:" + str(time_text))
    print("Total files found: ", len(f_lib))
    print("Total files found: ", len(timestamp))

    # Number of waveforms saved
    nr_channels = 3
    # Number of samples for each waveform
    nr_samples = 15000

    dataset = np.zeros((len(f_lib), nr_channels, nr_samples))
    metadata = np.zeros((len(f_lib), 3))

    for i in range(len(f_lib)):
        try:
            print("File " + str(i) + "of" + str(len(f_lib)), end="\r")

            with open(f_lib[i]) as f:
                data = json.load(f)
                dataset[i, 0, :] = data["ds"]
            with open(f_lib[i].replace("flux", "vibx")) as f:
                dataset[i, 1, :] = data["ds"]
            with open(f_lib[i].replace("flux", "vibz")) as f:
                dataset[i, 2, :] = data["ds"]
            with open(f_lib[i].replace("flux", "performance")) as f:
                data = json.load(f)
            for j, device in enumerate(devices):
                if device in f_lib[i]:
                    metadata[i, 0] = j
            metadata[i, 1] = data["rpm"]
            metadata[i, 2] = datetime.datetime.timestamp(timestamp[i])
        #
        except Exception as E:
            print("warning:" + f_lib[i])
    # print(dataset)
    # print(metadata)
    return dataset, metadata

def npy_creation_verdigris(path_to_data, devices, indexes):
    # Name of the sensors
    f_lib = []
    timestamp = []
    for device in devices:
        for filename in sorted(glob.glob(path_to_data + device + "*.gz")):
            try:
                f_lib.append(filename)
            except Exception as E:
                print("warning:" + filename)
    print("Total files found: ", len(f_lib))

    # Number of waveforms saved
    nr_channels = 3
    # Number of samples for each waveform
    nr_samples = 15000

    dataset = np.zeros((len(f_lib), nr_samples, nr_channels))
    metadata = np.zeros((len(f_lib), 3))
    fault_count = 0
    set_samp = {}
    for i in range(len(f_lib)):
        print(i, end="\r")
        with gzip.open(f_lib[i]) as f:
            try:
                j = 0
                for device in devices:
                    if device in f_lib[i]:
                        break
                    j += 1
                features = pd.read_csv(f, header=None)
                features = features.to_numpy()
                timestamp = features[0, -1] // 1000000000
                if features.shape[0] in set_samp:
                    set_samp[features.shape[0]] += 1
                else:
                    set_samp[features.shape[0]] = 1
                features = features[:15000, indexes[j]:indexes[j]+3]
                # features = np.transpose(features)
                dataset[i, :, :] = features
                metadata[i, 0] = j
                metadata[i, 1] = -1
                metadata[i, 2] = timestamp

            except Exception as E:
                # print(E)
                # print("warning:" + filename)
                fault_count += 1
    print("Faulty count",   fault_count)
    print(set_samp)
    dataset = np.moveaxis(dataset, 1, 2)

    return dataset, metadata


def plot_rpms(metadata):
    d = metadata[metadata[:, 1] > 0, 1]
    plt.hist(d, bins=np.arange(d.min(), d.max() + 1))
    plt.show()

def plot_samples_his():
    dic = {16896: 2744, 15360: 2454, 24064: 70, 13824: 35, 22528: 23, 7680: 119, 27136: 144, 9216: 31, 18432: 991, 25600: 18, 10752: 1, 21504: 19, 28672: 109, 26112: 5, 30208: 29, 19968: 54, 24576: 8, 23040: 21, 30720: 2, 31744: 5, 33280: 3, 32256: 1, 27648: 3, 29184: 1}
    k = list(dic.keys())
    k_s = [str(x) for x in sorted(k)]
    val = [x for _, x in sorted(zip(k, list(dic.values())))]
    plt.bar(k_s, val, color='g')
    plt.xticks(k_s[::2], rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.show()