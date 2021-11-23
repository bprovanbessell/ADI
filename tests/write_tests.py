import unittest
from influx.query_to_dataset import get_tm_interval
from influx.raw_data_to_infux import format_dt_ns
import numpy as np
import time
from influxdb_client import Point, InfluxDBClient, WriteOptions
import datetime
from influx.raw_data_to_infux import make_tm_points_batch


class WriteTest(unittest.TestCase):
    """
    Upload a single sample
    retrieve it
    check it is the same
    """

    def test_write_read(self):
        url = "http://localhost:8086"
        token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
        org = "Insight"
        bucket = "tests"

        write_dict = {"url": url,
                      "token": token,
                      "org": org,
                      "bucket": bucket}

        path_to_data = "../../time-series-influx/TS/data/test-motors/data-31-08-2021/"

        motors_data = np.load(path_to_data + "np_dataset.npy")
        motors_metadata = np.load(path_to_data + "np_metadata.npy")

        sample = motors_data[0]
        samplemd = motors_metadata[0]

        dt = datetime.datetime.fromtimestamp(int(samplemd[2]))

        dt2 = datetime.datetime.fromtimestamp(time.mktime(time.strptime("01.08.2021 02:00:00", "%d.%m.%Y %H:%M:%S")))

        print(dt)
        print(dt2)

        newstart = dt2 + datetime.timedelta(minutes=20)

        startstr = str(int(dt2.timestamp()))
        newstartstr = str(int(newstart.timestamp()))
        machine_id = int(samplemd[0])

        print(machine_id)

        columns = ["flux", "vibx", "vibz"]
        columns_str = "["
        for c in columns:
            columns_str += '"' + c + '", '
        columns_str += "]"

        tm_name = "test_motor_" + str(machine_id)

        with InfluxDBClient(url=url, token=token, org=org) as client:
            with client.write_api(write_options=WriteOptions(batch_size=15_000, flush_interval=5000)) as write_api:

                list_of_points = make_tm_points_batch(15000, dt, sample, machine_id)
                metadata_point = Point("test_motor_" + str(machine_id) + "_metadata") \
                    .tag("machine_id", str(machine_id)) \
                    .field("rpm", float(samplemd[1])) \
                    .time(format_dt_ns(dt, 0, 0))

                # write_api.write(bucket=bucket, record=list_of_points)
                # write_api.write(bucket=bucket, record=[metadata_point])


#         now get the interval
            interval = get_tm_interval(client, startstr, newstartstr, tm_name, columns, columns_str, 0, bucket)

            sample = np.moveaxis(sample, 0, 1)
            print(interval)
            print(sample)

            self.assertEqual(sample.shape, interval.shape)
            np.testing.assert_allclose(sample, interval)

