import glob
import datetime
import re
import csv
import json
import time

import numpy as np
import gzip
import pandas as pd
from influxdb_client import Point, InfluxDBClient, WriteOptions
import otosense_api

"""
Write the compressed files to the influx database
"""

'''
def otosense_influx_write_api_cont(devices, device_id, start_time, end_time, write_threshold, write_dict):
    """
    We need to sync the timestamps for all of the data
    
    One option is to get all of it first, then iterate through
    """
    influx_url = write_dict["url"]
    influx_token = write_dict["token"]
    influx_org = write_dict["org"]
    influx_bucket = write_dict["bucket"]

    datasets = ["vibx", "vibz", "flux", "tempe", "tempm", "performance", "conditions", "operations", "vibxFFT",
                "vibzFFT", "fluxFFT"]

    vibx_dataset = datasets[0]
    vibz_dataset = datasets[1]
    flux_dataset = datasets[2]
    performance_dataset = datasets[5]

    with InfluxDBClient(url=influx_url, token=influx_token, org=influx_org) as client:
        with client.write_api(write_options=WriteOptions(batch_size=15_000, flush_interval=5000)) as write_api:

            oto_api = otosense_api.OtosenseApi()
        # for j, device in enumerate(devices):
            j = device_id
            device = devices[j]

            # Now, from start_time to end_time, get the 3 datasets, vibx, vibz, flux (each should contain up to 120,
            # so either chunk it manually, or use the continuation tokens...
            # Synchronize the time between them (not sure what assumptions we can make here)
            # Should also get the performance dataset, as that contains the rpm
            # Will also need exceptions in the case that there is no data

            # do iterations through time...
            # can do up to 8 hours at a time, so iterate by 8 hours

            start = datetime.datetime.fromtimestamp(start_time)
            end = datetime.datetime.fromtimestamp(end_time)

            oto_api.write_all_motors()
            motor_id = oto_api.get_motor_id(device)

            list_of_points = []
            metadata_points = []

            nr_samples = 15000
            nr_channels = 3
            
            cont_token_flux = ""
            cont_token_vibx = ""
            cont_token_vibz = ""
            cont_token_performance = ""
            
            # The timestamps on the data might not be synced accordingly

            while cont_token is not None:

                # might have to add the utc offset here...
                iso_start = start.strftime('%Y-%m-%dT%H:%M:%SZ')
                iso_end = end.strftime('%Y-%m-%dT%H:%M:%SZ')

                print("start interval", iso_start)
                print("end interval", end)

                # make the api queries
                
                vibx_json = oto_api.get_data_continuation(motor_id, vibx_dataset, iso_start, iso_end, )
                vibz_json = oto_api.get_data_continuation(motor_id, vibz_dataset, iso_start, iso_end)
                flux_json = oto_api.get_data_continuation(motor_id, flux_dataset, iso_start, iso_end)
                performance_json = oto_api.get_data_continuation(motor_id, performance_dataset, iso_start, iso_end)

                # Sync timestamps somehow - future version
                # For each of the common timestamps
                # get the vibx, vibz, flux, rpm data
                rpm_time_data = oto_api.get_rpm_data(performance_json)
                vibx_time_data = oto_api.get_samples(vibx_json)
                vibz_time_data = oto_api.get_samples(vibz_json)
                flux_time_data = oto_api.get_samples(flux_json)

                # Add all samples found to the list of points
                for k in range(len(rpm_time_data)):
                    rpm_timestamp = rpm_time_data[k][1]
                    vibx_timestamp = vibx_time_data[k][1]
                    vibz_timestamp = vibz_time_data[k][1]
                    flux_timestamp = flux_time_data[k][1]

                    dataset = np.zeros((nr_channels, nr_samples))
                    dataset[0, :] = flux_time_data[k][0]
                    dataset[1, :] = vibx_time_data[k][0]
                    dataset[2, :] = vibz_time_data[k][0]

                    print("lengths of datasets")
                    print(len(flux_time_data[k][0]), len(vibx_time_data[k][0]), len(vibz_time_data[k][0]))

                    # timestamps should all be the same... Probably good to make a check in the future

                    dt = datetime.datetime.fromtimestamp(int(rpm_timestamp) // 1000)

                    points = make_tm_points_batch(nr_samples, dt, dataset, j)
                    list_of_points.extend(points)

                    metadata_point = Point("test_motor_" + str(j) + "_metadata") \
                        .tag("machine_id", str(j)) \
                        .field("rpm", float(rpm_time_data[k][0])) \
                        .time(format_dt_ns(dt, 0, 0))

                    metadata_points.append(metadata_point)

                # write the points to influx
                if len(list_of_points) >= write_threshold:
                    write_api.write(bucket=influx_bucket, record=list_of_points)
                    write_api.write(bucket=influx_bucket, record=metadata_points)

                    list_of_points = []
                    metadata_points = []

                # update the start time, to get the next interval
                start = newstart

            print("we done here")

        client.close()
'''

def otosense_influx_write_api(devices, device_id, start_time, end_time, write_threshold, write_dict):
    influx_url = write_dict["url"]
    influx_token = write_dict["token"]
    influx_org = write_dict["org"]
    influx_bucket = write_dict["bucket"]

    datasets = ["vibx", "vibz", "flux", "tempe", "tempm", "performance", "conditions", "operations", "vibxFFT",
                "vibzFFT", "fluxFFT"]

    vibx_dataset = datasets[0]
    vibz_dataset = datasets[1]
    flux_dataset = datasets[2]
    performance_dataset = datasets[5]

    with InfluxDBClient(url=influx_url, token=influx_token, org=influx_org) as client:
        with client.write_api(write_options=WriteOptions(batch_size=15_000, flush_interval=5000)) as write_api:

            oto_api = otosense_api.OtosenseApi()
        # for j, device in enumerate(devices):
            j = device_id
            device = devices[j]

            # Now, from start_time to end_time, get the 3 datasets, vibx, vibz, flux (each should contain up to 120,
            # so either chunk it manually, or use the continuation tokens...
            # Synchronize the time between them (not sure what assumptions we can make here)
            # Should also get the performance dataset, as that contains the rpm
            # Will also need exceptions in the case that there is no data

            # do iterations through time...
            # can do up to 8 hours at a time, so iterate by 8 hours

            start = datetime.datetime.fromtimestamp(start_time)
            end = datetime.datetime.fromtimestamp(end_time)

            oto_api.write_all_motors()
            motor_id = oto_api.get_motor_id(device)

            list_of_points = []
            metadata_points = []

            nr_samples = 15000
            nr_channels = 3

            while start < end:
                # next 20 minute interval
                # Go from start to newstart

                # write new version at some point that uses the continuation tokens,
                # as the api might change the internals
                newstart = start + datetime.timedelta(hours=8)

                # might have to add the utc offset here...
                iso_start = start.strftime('%Y-%m-%dT%H:%M:%SZ')
                iso_nstart = newstart.strftime('%Y-%m-%dT%H:%M:%SZ')

                print("start interval", iso_start)
                print("end interval", iso_nstart)

                # make the api queries
                vibx_json = oto_api.get_data(motor_id, vibx_dataset, iso_start, iso_nstart)
                vibz_json = oto_api.get_data(motor_id, vibz_dataset, iso_start, iso_nstart)
                flux_json = oto_api.get_data(motor_id, flux_dataset, iso_start, iso_nstart)
                performance_json = oto_api.get_data(motor_id, performance_dataset, iso_start, iso_nstart)

                # Sync timestamps somehow - future version
                # For each of the common timestamps
                # get the vibx, vibz, flux, rpm data
                rpm_time_data = oto_api.get_rpm_data(performance_json)
                vibx_time_data = oto_api.get_samples(vibx_json)
                vibz_time_data = oto_api.get_samples(vibz_json)
                flux_time_data = oto_api.get_samples(flux_json)

                # Add all samples found to the list of points
                for k in range(len(rpm_time_data)):
                    rpm_timestamp = rpm_time_data[k][1]
                    vibx_timestamp = vibx_time_data[k][1]
                    vibz_timestamp = vibz_time_data[k][1]
                    flux_timestamp = flux_time_data[k][1]

                    dataset = np.zeros((nr_channels, nr_samples))
                    dataset[0, :] = flux_time_data[k][0]
                    dataset[1, :] = vibx_time_data[k][0]
                    dataset[2, :] = vibz_time_data[k][0]

                    print("lengths of datasets")
                    print(len(flux_time_data[k][0]), len(vibx_time_data[k][0]), len(vibz_time_data[k][0]))

                    dt = datetime.datetime.fromtimestamp(int(rpm_timestamp) // 1000)

                    dtf = datetime.datetime.fromtimestamp(int(flux_timestamp) // 1000)
                    dtx = datetime.datetime.fromtimestamp(int(vibx_timestamp) // 1000)
                    dtz = datetime.datetime.fromtimestamp(int(vibz_timestamp) // 1000)
                    print("timestamps")
                    print(dt, dtf, dtx, dtz)

                    points = make_tm_points_batch(nr_samples, dt, dataset, j)
                    list_of_points.extend(points)

                    metadata_point = Point("test_motor_" + str(j) + "_metadata") \
                        .tag("machine_id", str(j)) \
                        .field("rpm", float(rpm_time_data[k][0])) \
                        .time(format_dt_ns(dt, 0, 0))

                    metadata_points.append(metadata_point)

                # write the points to influx
                if len(list_of_points) >= write_threshold:
                    write_api.write(bucket=influx_bucket, record=list_of_points)
                    write_api.write(bucket=influx_bucket, record=metadata_points)

                    list_of_points = []
                    metadata_points = []

                # update the start time, to get the next interval
                start = newstart

            # end while start < end

            write_api.write(bucket=influx_bucket, record=list_of_points)
            write_api.write(bucket=influx_bucket, record=metadata_points)
            print("we done here")

        client.close()


def otosense_influx_write_new(path_to_data, devices, write_threshold, write_dict):

    url = write_dict["url"]
    token = write_dict["token"]
    org = write_dict["org"]
    bucket = write_dict["bucket"]

    with InfluxDBClient(url=url, token=token, org=org) as client:
        with client.write_api(write_options=WriteOptions(batch_size=15_000, flush_interval=5000)) as write_api:
            # for each device/machine create a measurement

            # mew format contains a file for each machine, and each line within that file represents a 2 second sample
            for j, device in enumerate(devices):
                filename = glob.glob(path_to_data + "*" + device + "*")[0]
                print(filename)

                nr_channels = 3
                nr_samples = 15000

                list_of_points = []
                metadata_points = []

                i = 0
                try:
                    with open(filename) as f:
                        for line in f:
                            i += 1
                            print("Line " + str(i), end="\r")
                            data = eval(line)
                            # print(data.keys())
                            dt = datetime.datetime.fromtimestamp(int(data["timestamp"]) // 1000)

                            dataset = np.zeros((nr_channels, nr_samples))
                            dataset[0, :] = data["flux"]
                            dataset[1, :] = data["vibx"]
                            dataset[2, :] = data["vibz"]

                            points = make_tm_points_batch(nr_samples, dt, dataset, j)
                            list_of_points.extend(points)

                            # no rpm in this data format
                            metadata_point = Point("test_motor_" + str(j) + "_metadata") \
                                .tag("machine_id", str(j)) \
                                .field("rpm", float(0)) \
                                .time(format_dt_ns(dt, 0, 0))

                            metadata_points.append(metadata_point)

                            if len(list_of_points) >= write_threshold:

                                write_api.write(bucket=bucket, record=list_of_points)
                                write_api.write(bucket=bucket, record=metadata_points)

                                list_of_points = []
                                metadata_points = []

                except Exception as E:
                    print(E)
                    print("warning:" + filename)

                # write all final points
                write_api.write(bucket=bucket, record=list_of_points)
                write_api.write(bucket=bucket, record=metadata_points)

        client.close()


def otosense_influx_write(path_to_data, devices, write_threshold, write_dict):

    url = write_dict["url"]
    token = write_dict["token"]
    org = write_dict["org"]
    bucket = write_dict["bucket"]

    with InfluxDBClient(url=url, token=token, org=org) as client:
        with client.write_api(write_options=WriteOptions(batch_size=15_000, flush_interval=5000)) as write_api:

            # for each device/machine create a measurement
            for j, device in enumerate(devices):
                # get the list of files to generate the data from
                f_lib, timestamp = get_files_list_tm(path_to_data, device)

                nr_channels = 3
                nr_samples = 15000

                list_of_points = []
                metadata_points = []

                # test config should be 100 sets of 15 points
                # for i in range(100):
                for i in range(len(f_lib)):

                    dataset = np.zeros((nr_channels, nr_samples))
                    print("File " + str(i) + "of" + str(len(f_lib)), end="\r")

                    try:
                        print("File " + str(i) + "of" + str(len(f_lib)), end="\r")

                        with open(f_lib[i]) as f:
                            data = json.load(f)
                            dataset[0, :] = data["ds"]
                        with open(f_lib[i].replace("flux", "vibx")) as f:
                            data = json.load(f)
                            dataset[1, :] = data["ds"]
                        with open(f_lib[i].replace("flux", "vibz")) as f:
                            data = json.load(f)
                            dataset[2, :] = data["ds"]

                        dt = timestamp[i]

                        points = make_tm_points_batch(nr_samples, dt, dataset, j)

                        list_of_points.extend(points)

                        # for metadata only, here only the timestamp is needed
                        # keep if other info is necessary
                        with open(f_lib[i].replace("flux", "performance")) as f:
                            data = json.load(f)

                            metadata_point = Point("test_motor_" + str(j) + "_metadata") \
                                .tag("machine_id", str(j)) \
                                .field("rpm", float(data["rpm"])) \
                                .time(format_dt_ns(dt, 0, 0))

                            metadata_points.append(metadata_point)

                        # when there are a number of points above the threshold, write them to the db
                        if len(list_of_points) >= write_threshold:

                            write_api.write(bucket=bucket, record=list_of_points)
                            write_api.write(bucket=bucket, record=metadata_points)

                            list_of_points = []
                            metadata_points = []

                    except Exception as E:
                        print(E)
                        print("warning:" + f_lib[i])

                # write all final points
                write_api.write(bucket=bucket, record=list_of_points)
                write_api.write(bucket=bucket, record=metadata_points)

    client.close()


def otosense_influx_write_old(path_to_data, devices, write_threshold, write_dict):
    url = write_dict["url"]
    token = write_dict["token"]
    org = write_dict["org"]
    bucket = write_dict["bucket"]

    with InfluxDBClient(url=url, token=token, org=org) as client:
        with client.write_api(write_options=WriteOptions(batch_size=15_000, flush_interval=5000)) as write_api:

            # for each device/machine create a measurement
            for j, device in enumerate(devices):
            # for j in range(1, 3):
                device = devices[j]
                # get the list of files to generate the data from
                f_lib, timestamp = get_files_list_tm_old(path_to_data, device)

                nr_channels = 3
                nr_samples = 15000

                list_of_points = []
                metadata_points = []

                # test config should be 100 sets of 15 points
                # for i in range(100):
                for i in range(len(f_lib)):

                    dataset = np.zeros((nr_channels, nr_samples))
                    print("File " + str(i) + "of" + str(len(f_lib)), end="\r")

                    try:
                        print("File " + str(i) + "of" + str(len(f_lib)), end="\r")
                        with open(f_lib[i]) as f:
                            data = json.load(f)
                            dataset[0, :] = data["flux"]
                            dataset[1, :] = data["vibx"]
                            dataset[2, :] = data["vibz"]

                        dt = timestamp[i]

                        points = make_tm_points_batch(nr_samples, dt, dataset, j)

                        list_of_points.extend(points)

                        # for metadata only, here only the timestamp is needed
                        # keep if other info is necessary
                        with open(f_lib[i].replace("completeSample", "performance")) as f:
                            data = json.load(f)
                            metadata_point = Point("test_motor_" + str(j) + "_metadata") \
                                .tag("machine_id", str(j)) \
                                .field("rpm", float(data["rpm"])) \
                                .time(format_dt_ns(dt, 0, 0))

                            metadata_points.append(metadata_point)

                        # when there are a number of points above the threshold, write them to the db
                        if len(list_of_points) >= write_threshold:
                            write_api.write(bucket=bucket, record=list_of_points)
                            write_api.write(bucket=bucket, record=metadata_points)

                            list_of_points = []
                            metadata_points = []

                    except Exception as E:
                        print(E)
                        print("warning:" + f_lib[i])

                # write all final points
                write_api.write(bucket=bucket, record=list_of_points)
                write_api.write(bucket=bucket, record=metadata_points)

    client.close()


def get_files_list_tm_old(path_to_data, device):
    f_lib = []
    timestamp = []
    for filename in sorted(glob.glob(path_to_data + device + "/completeSample/data*")):
        try:
            f_lib.append(filename)
            timestamp.append(
                datetime.datetime.fromtimestamp(int(re.split("[# .]", filename)[-2]) / 1000)
            )
        except Exception as E:
            print("warning:" + filename)
            print(E)
    print("Total files found: ", len(f_lib))

    return f_lib, timestamp


def get_files_list_tm(path_to_data, device):
    f_lib = []
    timestamp = []

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
                    print("Missing vib:" + filename)
            except Exception as E:
                print("warning:" + filename + "\nTime:" + str(time_text))
                print(E)
    print("Total files found: ", len(f_lib))
    print("Total files found: ", len(timestamp))

    return f_lib, timestamp


def make_tm_points_batch(nr_samples, init_dt, dataset, machine_id):
    increment = int(1 / 7500 * 10 ** 9)
    base_ns = 0
    second = 0
    lines = []

    for j in range(nr_samples):
        # change init_timestamp when j past 7500
        if j == 7500:
            # add extra second to datetime
            init_dt = init_dt + datetime.timedelta(0, 1)
            # reset nanoseconds to 0
            base_ns = 0
            second = 1

        ndt = format_dt_ns(init_dt, base_ns, second)
        # if(j == 0):
        #     print(ndt)
        p = Point("test_motor_" + str(machine_id)) \
            .field("flux", float(dataset[0][j])) \
            .field("vibx", float(dataset[1][j])) \
            .field("vibz", float(dataset[2][j])) \
            .time(ndt)
        lines.append(p)

        base_ns += increment

    return lines


def verdigris_influx_write(path_to_data, devices, indexes, write_threshold, write_dict):

    url = write_dict["url"]
    token = write_dict["token"]
    org = write_dict["org"]
    bucket = write_dict["bucket"]

    with InfluxDBClient(url=url, token=token, org=org) as client:
        with client.write_api(write_options=WriteOptions(batch_size=15_000, flush_interval=5000)) as write_api:

            # Name of the sensors
            for j, device in enumerate(devices):
                f_lib = get_files_list_ver(path_to_data, device)

                # Number of waveforms saved
                nr_channels = 3
                # Number of samples for each waveform, in this case we want to write all of the data
                # nr_samples = 15000
                list_of_points = []

                fault_count = 0
                set_samp = {}
                for i in range(len(f_lib)):
                # for i in range(3):
                    print(i, end="\r")
                    with gzip.open(f_lib[i]) as f:
                        try:
                            features = pd.read_csv(f, header=None)
                            features = features.to_numpy()
                            timestamp = features[0, -1] // 1000000000
                            if features.shape[0] in set_samp:
                                set_samp[features.shape[0]] += 1
                            else:
                                set_samp[features.shape[0]] = 1

                            # only takes the first 15000k values
                            # features = features[:15000, indexes[j]:indexes[j] + 3]

                            features = features[:, indexes[j]:indexes[j] + 3]
                            # upload all the values, will have to query them properly later
                            # print("features shape", features.shape)
                            dt = datetime.datetime.fromtimestamp(timestamp)

                            points = make_ver_points_batch(dt, features, j)

                            list_of_points.extend(points)

                            if len(list_of_points) >= write_threshold:
                                write_api.write(bucket=bucket, record=list_of_points)

                                list_of_points = []

                        except Exception as E:
                            print(E)
                            # print("warning:" + filename)
                            fault_count += 1

                # write any final points
                write_api.write(bucket=bucket, record=list_of_points)


def make_ver_points_batch(init_dt, dataset, machine_id):

    increment = int(1 / 7500 * 10 ** 9)
    base_ns = 0
    second = 0
    lines = []

    for j in range(dataset.shape[0]):
        # change init_timestamp when j past 7500
        if j % 7500 == 0 and j != 0:
            # add extra second to datetime
            init_dt = init_dt + datetime.timedelta(0, 1)
            # reset nanoseconds to 0
            base_ns = 0
            second += 1

        ndt = format_dt_ns(init_dt, base_ns, second)
        # probably possible to optimise by writing multiple rows
        # row = [dataset[j][0], dataset[j][1], dataset[j][2], ndt]

        p = Point("verdigris_" + str(machine_id)) \
            .field("current0", float(dataset[j][0])) \
            .field("current1", float(dataset[j][1])) \
            .field("current2", float(dataset[j][2])) \
            .time(ndt)
        lines.append(p)
        # writer.writerow(row)
        base_ns += increment

    return lines


def get_files_list_ver(path_to_data, device):
    f_lib = []
    for filename in sorted(glob.glob(path_to_data + device + "*.gz")):
        try:
            f_lib.append(filename)
        except Exception as E:
            print("warning:" + filename)
    print("Total files found: ", len(f_lib))

    return f_lib


def format_dt_ns(init_dt, ns, s):
    # round to nearest 20
    minute = round_minutes(init_dt.minute)

    return '{}:{:02.0f}:0{}.{:09.0f}Z'.format(init_dt.strftime('%Y-%m-%dT%H'), minute, s, ns)


def round_minutes(mins):
    rounded = 20 * round(mins / 20)

    if rounded == 60:
        return 0
    else:
        return rounded


if __name__ == "__main__":

    url = "http://localhost:8086"

    # local machine token
    token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="

    org = "Insight"

    # bucket = "test_bucket"
    bucket = "ADI"

    write_dict = {"url": url,
                  "token": token,
                  "org": org,
                  "bucket": bucket}

    write_threshold = 150000

    # Path of the Otosense/TestMotors data
    # path_to_data = "/Volumes/Elements/ADI/data_tm20/"
    path_to_tm_data = "../data/test-motors/data-31-08-2021/"

    # Machine names
    tm_devices = ["BlockAScrubber", "PU7001", "GeneralCoolingLoop"]
    tm_devices_new = ["Block-A-Scrubber", "PU7001", "General_Cooling_Loop"]

    path_to_tm_data_new = "../../all_data/Dec data/data-31-10-2021/"

    otosense_influx_write_new(path_to_tm_data_new, tm_devices_new, write_threshold, write_dict)

    # otosense_influx_write(path_to_tm_data, tm_devices, write_threshold, write_dict)

    # verdigris_influx_write(path_to_ver_data, ver_devices, indexes, write_threshold, write_dict)
