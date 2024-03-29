from util import dataset
from configurations import ds_config
import numpy as np
import datetime
import time
from backports.zoneinfo import ZoneInfo
from influxdb_client import InfluxDBClient, Point, WritePrecision


"""
For the time and test intervals

For each 20 minute window, check if the window is valid (e.g. has a high enough rpm, and has 15,000 points)
Get the window, and set it the np array dataset
Keep track of the total number of valid windows

Trim the final dataset at the end to the final number of windows
"""


def query_to_dataset(start, end, signal, nr_sample, machine, speed_limit, write_dict):

    time_length = end - start
    twentymins = 60 * 20
    upper_intervals = int(time_length / twentymins)

    zone = ZoneInfo("Europe/Dublin")

    start = datetime.datetime.fromtimestamp(start, tz=zone)
    end = datetime.datetime.fromtimestamp(end, tz=zone)

    columns = []
    # get columns we want
    if signal == "flux":
        columns = ["flux"]
        channels = 1

    if signal == "vibration":
        columns = ["vibx", "vibz"]
        channels = 2

    if signal == "current":
        columns = ["current0", "current1", "current2"]
        channels = 3

    if signal == "all":
        columns = (["flux", "vibx", "vibz"], ["current0", "current1", "current2"])
        channels = 6

    if signal == "flux_vib":
        columns = ["flux", "vibx", "vibz"]
        channels = 3

    # set np array of zeros
    # upper limit on what it will be, reshaping later will be necessary
    np_dataset = np.zeros((upper_intervals, nr_sample, channels))

    np_meta_data = np.zeros((upper_intervals, 3))
    for u in range(upper_intervals):
        np_meta_data[u, 0] = machine

    print("init shape: ", np_dataset.shape)
#     query to get the 20 minute interval
#     write it to the dataset
    if signal == "all":
        columns_str = (get_columns_str(columns[0]), get_columns_str(columns[1]))

    else:
        columns_str = get_columns_str(columns)

    # current is verdigris, flux and vibration are test_motor
    if signal == "current":
        machine_name = "verdigris_" + str(machine)
    elif signal == "flux" or signal == "vibration" or signal == "flux_vib":
        machine_name = "test_motor_" + str(machine)
    else:
        machine_name = ["test_motor_" + str(machine), "verdigris_" + str(machine)]

    print(machine_name)

    valid_intervals = get_and_set_intervals(np_dataset, np_meta_data, start, end, machine_name, columns, columns_str, speed_limit, write_dict)
    # trim the dataset
    print(valid_intervals)
    np_dataset = np_dataset[:valid_intervals]
    np_meta_data = np_meta_data[:valid_intervals]

    print("Final dataset shape: ", np_dataset.shape)

    return np_dataset, np_meta_data


def get_and_set_intervals(np_dataset, np_meta_data, start, end, machine_name, columns, columns_str, speed_limit, write_dict):
    url = write_dict["url"]
    token = write_dict["token"]
    org = write_dict["org"]
    bucket = write_dict["bucket"]

    valid_intervals = 0

    with InfluxDBClient(url=url, token=token, org=org) as client:
        print("start: ", start)
        print("end: ", end)

        # inclusive, exclusive
        while start < end:
            # next 20 minute interval
            newstart = start + datetime.timedelta(minutes=20)

            # check if we are in non daylight savings, if that is the case, then increment start by 1 hour
            utcoffset = start.tzinfo.utcoffset(start)
            # 2 second intervals, 20 minutes apart
            interval_end = start + datetime.timedelta(seconds=2) + utcoffset
            interval_endstr = str(int(interval_end.timestamp()))

            start_stamp = int((start + utcoffset).timestamp())
            startstr = str(start_stamp)

            # newstart_stamp = int(newstart.timestamp())
            # newstartstr = str(newstart_stamp)

            if len(machine_name) == 2:
                # get both of the machine intervals
                ver_interval = get_ver_interval(client, startstr, interval_endstr, machine_name[1], columns[1], columns_str[1], bucket)
                tm_interval, rpm = get_tm_interval(client, startstr, interval_endstr, machine_name[0], columns[0], columns_str[0],
                                                speed_limit, bucket)

                if is_valid_interval(ver_interval) and is_valid_interval(tm_interval):
                    np_dataset[valid_intervals, :, 0:3] = tm_interval[:15000]
                    np_dataset[valid_intervals, :, 3:6] = ver_interval[:15000]

                    # due to daylight savings, so if it is between 28 March and 31 October, then this timestamp should be decremented by 1 hour
                    np_meta_data[valid_intervals, 2] = start_stamp - utcoffset.seconds
                    np_meta_data[valid_intervals, 1] = rpm
                    valid_intervals += 1
                else:
                    print("Invalid Interval")
                    # print(ver_interval)
                    # print(tm_interval)

            else:

                if "verdigris" in machine_name:
                    interval = get_ver_interval(client, startstr, interval_endstr, machine_name, columns, columns_str, bucket)
                    rpm = -1
                else:
                    # flux or vib, or both
                    interval, rpm = get_tm_interval(client, startstr, interval_endstr, machine_name, columns, columns_str, speed_limit, bucket)

                if interval is not None:
                    # print("length: ", len(interval))
                    if len(interval) >= 15000:
                        np_dataset[valid_intervals] = interval[:15000]
                        # due to daylight savings, so if it is between 28 March and 31 October, then this timestamp should be decremented by 1 hour

                        np_meta_data[valid_intervals, 2] = start_stamp - utcoffset.seconds
                        np_meta_data[valid_intervals, 1] = rpm
                        valid_intervals += 1
                    else:
                        # np_dataset[valid_intervals, 0:len(interval)] = interval[0:len(interval)]
                        print("less than 15000")
                        print(interval.shape)
                    # valid_intervals += 1

            start = newstart

        client.close()

    return valid_intervals


def is_valid_interval(interval):
    if interval is not None:
        if len(interval) >= 15000:
            return True

    return False


def get_columns_str(columns):
    columns_str = "["
    for c in columns:
        columns_str += '"' + c + '", '
    columns_str += "]"
    return columns_str

"""
First get the metadata and make sure the rpm is high enough
Then get the window, convert it to np array, set the training dataset
"""


def get_tm_interval(client, start, end, tm_name, columns, columns_str, speed_limit, bucket):

    try:
        query = 'from(bucket:"' + bucket + '") ' \
                '|> range(start: ' + start + ', stop: ' + end + ') ' \
                '|> filter(fn: (r) => r._measurement == "' + tm_name + "_metadata" + '")'

        res = client.query_api().query(query)

        if len(res) > 0:

            rpm = res[0].records[0].get_value()
            # check is only necessary if speed limit greater than 0
            if rpm > speed_limit or speed_limit <= 0:
                # rpm is alright, fetch the points for this interval window
                query = 'from(bucket:"' + bucket + '") ' \
                        '|> range(start: ' + start + ', stop: ' + end + ') ' \
                        '|> filter(fn: (r) => r._measurement == "' + tm_name + '")' \
                        '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") ' \
                        '|> keep(columns: ' + columns_str + ')'

                data_frame = client.query_api().query_data_frame(query=query)
                cols = data_frame[columns]
                cols = cols.to_numpy()

                return cols, rpm

        return None, 0

    except Exception as E:
        print(E)
        print("Start: ", start)
        print("End  : ", end)
        return None, 0


def get_ver_interval(client, start, end, ver_name, columns, columns_str, bucket):

    query = 'from(bucket:"' + bucket + '") ' \
            '|> range(start: ' + start + ', stop: ' + end + ') ' \
            '|> filter(fn: (r) => r._measurement == "' + ver_name + '")' \
            '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value") ' \
            '|> keep(columns: ' + columns_str + ')'

    try:

        data_frame = client.query_api().query_data_frame(query=query)
        if not data_frame.empty:

            cols = data_frame[columns]
            cols = cols.to_numpy()

            return cols
        else:
            return None

    except Exception as E:
        print(E)
        print("Start: ", start)
        print("End  : ", end)
        return None


if __name__ == "__main__":

    # do a query test
    url = "http://localhost:8086"
    token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
    org = "Insight"
    bucket = "test_bucket"

    write_dict = {"url": url,
                  "token": token,
                  "org": org,
                  "bucket": bucket}

    test_dataset = dataset.Dataset

    # Don't need it for this test
    data_path = ""

    # will have to modify dataset class, so that it can be created from influx

    ds_config.DatasetConfiguration().SetConfiguration(test_dataset, data_path, 'EXa_1_Curr')
    # get the first day = 72 intervals
    test_dataset.time_train_start = time.mktime(time.strptime("01.08.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
    # Training time end
    test_dataset.time_train_end = time.mktime(time.strptime("02.08.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

    # duds for now
    test_dataset.time_test_start = time.mktime(time.strptime("01.08.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))
    test_dataset.time_test_end = time.mktime(time.strptime("01.08.2021 00:00:00", "%d.%m.%Y %H:%M:%S"))

    # machine is actually on here
    test_dataset.machine = 1
    test_dataset.nr_sample = 15000

    # test_dataset.signal = "vibration"
    test_dataset.signal = "current"

    test_time_length = test_dataset.time_train_end - test_dataset.time_train_start

    query_to_dataset(test_dataset, write_dict)
#     init should be 72
#     final should be <= 72, intervals start at 2am
