"""
https://adi.otosensesms.com/api-reference

"""

import requests
import json
import datetime
import time


# Do we have a specific api endpoint

class OtosenseApi:

    def __init__(self):
        self.start_time = time.time()
        self.endpoint = "https://toqg6279fi.execute-api.eu-west-1.amazonaws.com/prod/"
        self.path_base = "../"

        self.bearer_token = self.get_authentication_token()


    def get_otosense_credentials(self):

        with open(self.path_base + "api_files/otosense_cred.json", 'r') as cred_file:
            data = json.load(cred_file)

            return data["Client ID"], data["Client Secret"]

    def check_and_update_bearer_token(self):
        if time.time() - self.start_time < (self.valid_time - 10):
            # update the token
            print("Token out of date, update the bearer token")
            self.bearer_token = self.get_authentication_token()

    def get_authentication_token(self):

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

        for record in measurement_json["records"]:
            timestamp = record["timestamp"]
            meas_data = record["data"]

            meas_time_data.append((meas_data, timestamp))

        return meas_time_data


if __name__ == "__main__":
    # print(get_otosense_credentials())

    # Should be valid for an hour
    # get_authentication_token()

    # Set up pipeline
    # From start time to end time, get the 3 different datasets (vibx, vibz, flux), for each 3 different machines
    # Concatenate it together somehow into points

    # So, do 3 requests at the same time?, save each of the json results as dictionaries,
    # (don't even need to bother saving the result as a file, just go straight from the API to influx)

    # iterate through the 3 results, just checking to make sure that all the timestamps are the same (or within certain tolerance?)
    # create the influxdb points from it, upload
    # Keep track of the continuation tokens
    # bearer_token = read_bearer_token()

    # print(bearer_token)

    # get_all_motors(bearer_token)

    print("---------MOTOR----------")
    # get_specific_motor(bearer_token, motor_id3)

    # get_data(bearer_token, motor_id3)

