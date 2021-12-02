from influxdb_client import Point, InfluxDBClient, WriteOptions
import datetime

# Should have list of results, along with a list of timestamps for those results
# for certain machine as well


def write_results_to_influx(write_dict, list_of_points):
    url = write_dict["url"]
    token = write_dict["token"]
    org = write_dict["org"]
    bucket = write_dict["bucket"]

    print(list_of_points[0])

    with InfluxDBClient(url=url, token=token, org=org) as client:
        with client.write_api(write_options=WriteOptions(batch_size=5_000, flush_interval=5000)) as write_api:
            write_api.write(bucket=bucket, record=list_of_points)


def make_results_points(timestamps, results, model_name, dataset_name):

    datetimes = [format_dt(x) for x in timestamps]

    print(datetimes[0])
    print(results[0])

    print(len(datetimes), len(results))

    points = []

    for dt, result in zip(datetimes, results):

        p = Point(model_name + "/" + dataset_name) \
            .field("result", float(result)) \
            .time(dt)
        points.append(p)

    return points


def format_dt(timestamp):

    dt = datetime.datetime.fromtimestamp(int(timestamp))

    return dt.strftime('%Y-%m-%dT%H:%M:00.000Z')


