import time
import datetime
from backports.zoneinfo import ZoneInfo
from raw_data_to_infux import otosense_influx_write, verdigris_influx_write

if __name__ == "__main__":

    t0 = time.time()

    url = "http://localhost:8086"
    token = "xZQSsWwOWDRoVwN4fx0o78_ZUgDgGE15Gbllb4iunKYTb9mutrcX4fvapJ2AkAC8buGih0qopwaumkHzIUjWFA=="
    org = "Insight"
    bucket = "ADI"

    write_dict = {"url": url,
                  "token": token,
                  "org": org,
                  "bucket": bucket}

    batches = 50
    write_threshold = 15000 * batches

    # path_to_tm_data = "data/test-motors/data-31-08-2021/"

    # path_to_tm_data = "../../all_data/ADI/data_tm20/"
    path_to_tm_data = "../../../all_data/ADI/data_tm/"

    # Machine names
    # tm_devices = ["BlockAScrubber", "PU7001", "GeneralCoolingLoop"]
    tm_devices = ["Grundfoss", "PU7001", "GeneralCoolingLoop"]

    # path_to_ver_data = "data/verdigris/"

    # all the data
    path_to_ver_data = "../../../all_data/ADI/data_verdigris/"

    # Name of the motors (they are in the same order as the previous ones)
    ver_devices = ["JBE10001123", "JBE10001196", "JBE10001268"]
    # The numpy arrays contain many columns, the one of interest start at these indexes
    indexes = [6, 12, 9]

    # otosense_influx_write(path_to_tm_data, tm_devices, write_threshold, write_dict)

    # path_to_ver_data = "../../all_data/2021-05/"
    # verdigris_influx_write(path_to_ver_data, ver_devices, indexes, write_threshold, write_dict)
    #
    # path_to_ver_data = "../../all_data/2021-06/"
    # verdigris_influx_write(path_to_ver_data, ver_devices, indexes, write_threshold, write_dict)
    #
    # path_to_ver_data = "../../all_data/2021-07/"
    # verdigris_influx_write(path_to_ver_data, ver_devices, indexes, write_threshold, write_dict)

    t1 = time.time()

    zone = ZoneInfo("Europe/Dublin")

    x = 1616895600

    dt3 = datetime.datetime.fromtimestamp(x, tz=ZoneInfo("Europe/Dublin"))

    dt = datetime.datetime.fromtimestamp(t1, tz=ZoneInfo("Europe/Dublin"))
    dt2 = datetime.datetime(2021, 3, 28, 2, 0, 0, tzinfo=zone)
    print(zone)
    print(dt.tzinfo)
    print(dt.tzinfo.utcoffset(dt))
    print(type(dt3.tzinfo.utcoffset(dt3)))

    print(dt3)


    total = t1 - t0
    print("Total seconds is: ", total)
    print("minutes is: ", total/60)
