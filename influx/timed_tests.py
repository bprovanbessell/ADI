import time
import datetime
from backports.zoneinfo import ZoneInfo
from raw_data_to_infux import otosense_influx_write, verdigris_influx_write, otosense_influx_write_new, otosense_influx_write_api
from util import preprocessing
import sys
import os

if __name__ == "__main__":

    t0 = time.time()

    # url = "http://localhost:8086"
    # for the gpu ssh tunnel
    url = "http://localhost:9090"
    # token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
    # gpu token
    token = "sPxOPI2tfrYVVjpC3b8IwxtJnv8ISRmTr_rEDaX4Q6WDj_SA1TjXPpplR26oJwHFB9aIei07jhsqHXdXkT6VnQ=="
    org = "Insight"
    bucket = "ADI"

    write_dict = {"url": url,
                  "token": token,
                  "org": org,
                  "bucket": bucket}

    batches = 50
    write_threshold = 15000 * batches

    local_write_dict = {"url": "http://localhost:8086",
                  "token": "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA==",
                  "org": org,
                  "bucket": bucket}

    # path_to_tm_data = "../../all_data/sample_data/data-31-08-2021/"
    # path_to_tm_data = "../../all_data/ADI/data_tm/"
    # path_to_tm_data = "../../all_data/Dec data/data-31-05-2021/"

    # Machine names
    tm_devices = ["BlockAScrubber", "PU7001", "GeneralCoolingLoop"]
    # tm_devices = ["Grundfoss", "PU-7001", "GeneralCoolingLoop"]
    tm_devices_new = ["Block-A-Scrubber", "PU7001", "General_Cooling_Loop"]

    # all the data
    path_to_ver_data = "../../all_data/upload/"

    # Name of the motors (they are in the same order as the previous ones)
    ver_devices = ["JBE10001123", "JBE10001196", "JBE10001268"]
    # The numpy arrays contain many columns, the one of interest start at these indexes
    indexes = [6, 12, 9]

    path_to_tm_data = "../../all_data/otosense/data-31-11-2021/"
    # otosense_influx_write(path_to_tm_data, tm_devices, write_threshold, write_dict)
    # otosense_influx_write_new(path_to_tm_data, tm_devices_new, write_threshold, write_dict)

    # path_to_ver_data = "../../all_data/Dec data/verd new/"
    path_to_ver_data = "../verdigris_files/2022_files/"
    verdigris_influx_write(path_to_ver_data, ver_devices, indexes, write_threshold, write_dict)

    tm_devices_api = ["Block A Scrubber", "PU7001", "General Cooling Loop"]
    # General cooling loop is still online

    upload_start = time.mktime(time.strptime("01.03.2022 00:00:00", "%d.%m.%Y %H:%M:%S"))
    upload_end = time.mktime(time.strptime("01.03.2022 02:00:00", "%d.%m.%Y %H:%M:%S"))

    # print(os.getcwd())
    # print(sys.path)
    # otosense_influx_write_api(tm_devices_api, 2, upload_start, upload_end, write_threshold, local_write_dict)

    t1 = time.time()

    total = t1 - t0
    print("Minutes: ", total/60)
    print("Hours: ", total/(60*60))
