"""
https://adi.otosensesms.com/api-reference

"""

import requests
import json
from datetime import datetime, timedelta
import time
import numpy as np


# Do we have a specific api endpoint

class OtosenseApi:

    def __init__(self, verbose=False):
        self.start_time = time.time()
        self.endpoint = "https://toqg6279fi.execute-api.eu-west-1.amazonaws.com/prod/"
        self.path_base = "../"
        self.verbose = verbose

        self.bearer_token = self.get_authentication_token()

    def get_otosense_credentials(self):

        with open(self.path_base + "api_files/otosense_cred.json", 'r') as cred_file:
            data = json.load(cred_file)

            return data["Client ID"], data["Client Secret"]

    def check_and_update_bearer_token(self):
        if (time.time() - self.start_time) > (self.valid_time - 10):
            # update the token
            if self.verbose:
                print("Token out of date, update the bearer token")
            self.bearer_token = self.get_authentication_token()

    def get_authentication_token(self):
        if self.verbose:
            print("Getting bearer token")

        auth_ending = "oauth/token"

        url = self.endpoint + auth_ending

        application_json = {
            "grant_type": "client_credentials"
        }

        client_id, client_secret = self.get_otosense_credentials()

        r = requests.post(url=url, json=application_json, auth=(client_id, client_secret))

        print(r)

        res = r.json()

        b_file = open(self.path_base + "api_files/bearer_credentials.txt", "w")
        b_file.write(res["access_token"])

        self.valid_time = int(res["expires_in"])

        b_file.close()

        return res["access_token"]

    def read_bearer_token(self):
        b_file = open(self.path_base + "api_files/bearer_credentials.txt", "r")
        res = b_file.read().strip("\n")
        return res

    def write_all_motors(self):
        # https: // your - api - endpoint.otosensesms.com / motors
        headers={"Authorization": "Bearer " + self.bearer_token,
                 "Accepth-Encoding": "gzip, deflate, br"}
        url = self.endpoint + "motors"
        r = requests.get(url=url, headers=headers)

        print(r)

        motors_file = open(self.path_base + "api_files/motors_cred.json", "w")
        json.dump(r.json(), motors_file)

    def get_specific_motor(self, motor_id):
        # https: // your - api - endpoint.otosensesms.com / motors
        headers={"Authorization": "Bearer " + self.bearer_token,
                 "Accepth-Encoding": "gzip, deflate, br"}
        url = self.endpoint + "motors" + "/" + motor_id
        r = requests.get(url=url, headers=headers)

        print(r)
        print(r.json())

    def get_data(self, motor_id, dataset, start, end):
        if self.verbose:
            print("getting data from: ", start, " to: ", end, " from motor: ", motor_id)

        # Make sure the bearer token is still valid
        self.check_and_update_bearer_token()
        # Seems to be up to 120 sets of records for 1 request
        # But that might be different depending on the dataset

        headers = {"Authorization": "Bearer " + self.bearer_token,
                   "Accepth-Encoding": "gzip, deflate, br"}

        # https: // your - api - endpoint.otosensesms.com / data / {motorId} / {dataset}
        url = self.endpoint + "data/" + motor_id + "/" + dataset

        # ISO-8601 date and time
        # '2022-03-01T11:23:19.715Z'
        # start = "2021-01-01T00:00:00Z"
        # end = "2022-02-06T13:38:14Z"
        payload = {'start': start, 'end': end}

        r = requests.get(url=url, headers=headers, params=payload)

        print(r)
        # res_file = open("documentation/results2.json", "w")
        # json.dump(r.json(), res_file)
        return r.json()

    def get_data_continuation(self, motor_id, dataset, start, end, cont_token):
        # Make sure the bearer token is still valid
        self.check_and_update_bearer_token()
        # Seems to be up to 120 sets of records for 1 request
        # But that might be different depending on the dataset

        headers = {"Authorization": "Bearer " + self.bearer_token,
                   "Accepth-Encoding": "gzip, deflate, br"}

        # https: // your - api - endpoint.otosensesms.com / data / {motorId} / {dataset}
        url = self.endpoint + "data/" + motor_id + "/" + dataset

        if cont_token == "":
            payload = {'start': start, 'end': end}
        else:
            payload = {'start': start, 'end': end, "continuationToken": cont_token}

        r = requests.get(url=url, headers=headers, params=payload)

        # There should be some error catching here, if there is no data to pull
        print(r)

        return r.json()

    def get_motor_id(self, motor_name):

        motors_file = open(self.path_base + "api_files/motors_cred.json")
        d = json.load(motors_file)

        for motor in d["motors"]:
            motor_id = motor["motorId"]
            if motor["attributes"]["name"] == motor_name:
                return motor_id

        raise NameError("motor name not found")

    def get_rpm_data(self, performance_json):
        rpm_time_data = []

        for record in performance_json["records"]:
            timestamp = record["timestamp"]
            rpm = record["rpm"]

            rpm_time_data.append((rpm, timestamp))

        return rpm_time_data

    def get_samples(self, measurement_json):

        meas_time_data = []

        try:
            for record in measurement_json["records"]:
                timestamp = record["timestamp"]
                meas_data = record["data"]

                meas_time_data.append((meas_data, timestamp))

            return meas_time_data
        except KeyError:
            print("No records in this dataset")
            return None

    def get_cont_token(self, measurement_json):
        try:
            cont_token = measurement_json["continuationToken"]
        except KeyError:
            cont_token = None
        return cont_token

    def get_single_sample(self, dt: datetime, device_id, motor_name):
        start_dt = dt
        end_dt = dt + timedelta(minutes=20)
        iso_start = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        iso_end = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        nr_channels = 15000
        nr_samples = 3
        dataset = np.zeros((nr_channels, nr_samples))

        motor_id = self.get_motor_id(motor_name)

        vibx_json = self.get_data(motor_id, "vibx", iso_start, iso_end)
        vibz_json = self.get_data(motor_id, "vibz", iso_start, iso_end)
        flux_json = self.get_data(motor_id, "flux", iso_start, iso_end)
        performance_json = self.get_data(motor_id, "performance", iso_start, iso_end)

        rpm_time_data = self.get_rpm_data(performance_json)
        vibx_time_data = self.get_samples(vibx_json)
        vibz_time_data = self.get_samples(vibz_json)
        flux_time_data = self.get_samples(flux_json)

        # Should check all of the samples.., although I would assume if one is missing then the rest will be aswell
        if not all([vibx_time_data, vibz_time_data, flux_time_data]):
            print("missing some data")
            return None

        flux_data = [float(x) for x in flux_time_data[0][0]]
        vibx_data = [float(x) for x in vibx_time_data[0][0]]
        vibz_data = [float(x) for x in vibz_time_data[0][0]]

        dataset[:, 0] = flux_data
        dataset[:, 1] = vibx_data
        dataset[:, 2] = vibz_data

        return dataset


if __name__ == "__main__":
    api = OtosenseApi()

    start = time.mktime(time.strptime("17.01.2022 00:00:00", "%d.%m.%Y %H:%M:%S"))
    end = time.mktime(time.strptime("17.01.2022 01:00:00", "%d.%m.%Y %H:%M:%S"))

    start_dt = datetime.fromtimestamp(start)
    end_dt = datetime.fromtimestamp(end)

    iso_start = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    iso_end = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    tm_devices_api = ["Block A Scrubber", "PU7001", "General Cooling Loop"]

    motor_id0 = api.get_motor_id(tm_devices_api[0])
    motor_id1 = api.get_motor_id(tm_devices_api[1])
    motor_id2 = api.get_motor_id(tm_devices_api[2])

    # res = api.get_data(motor_id0, "flux", iso_start, iso_end)
    # res = api.get_data(motor_id1, "flux", iso_start, iso_end)
    # print(res)

    print("rpm")

    start = time.mktime(time.strptime("17.01.2022 00:00:00", "%d.%m.%Y %H:%M:%S"))
    end = time.mktime(time.strptime("17.01.2022 00:30:00", "%d.%m.%Y %H:%M:%S"))

    start_dt = datetime.fromtimestamp(start)
    end_dt = datetime.fromtimestamp(end)

    iso_start = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    iso_end = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    perf = api.get_data(motor_id2, "performance", iso_start, iso_end)
    res = api.get_data(motor_id2, "flux", iso_start, iso_end)

    print(len(perf["records"]))
    print(len(res["records"][0]["data"]))

    # length of records is based on size, so flux is capped at 120 records, where performance cap is much higher
