"""https://adi.otosensesms.com/api-reference


Base64 encode client credentials
get the bearer token, response should signify how long it is acceptable for

Provide the valid bearer token in the http header
"""

import requests
import json
import datetime
endpoint = "https://toqg6279fi.execute-api.eu-west-1.amazonaws.com/prod/"


# Do we have a specific api endpoint

def get_otosense_credentials():

    with open("documentation/otosense_cred.json", 'r') as cred_file:
        data = json.load(cred_file)

        return data["Client ID"], data["Client Secret"]


def get_authentication_token():

    auth_ending = "oauth/token"

    url = endpoint + auth_ending

    application_json = {
        "grant_type": "client_credentials"
    }

    # header field shoukd be in the form of
    # Authorization: Basic {credentials} where credentials is the Base64 encoding of client_id and client_secret joined by a single colon :.

    client_id, client_secret = get_otosense_credentials()

    r = requests.post(url=url, json=application_json, auth=(client_id, client_secret))

    print(r)

    # print(r.json())

    res = r.json()

    # probably good to sort out timing as well, should set it up so we also know when the token is invalid, probably store it in a file aswell, so that it is never committed

    b_file = open("documentation/bearer_credentials.txt", "w")
    b_file.write(res["access_token"])
    b_file.close()

    return res["access_token"]


def read_bearer_token():
    b_file = open("documentation/bearer_credentials.txt", "r")
    res = b_file.read().strip("\n")
    return res


def get_all_motors(bearer_token):
    # https: // your - api - endpoint.otosensesms.com / motors
    headers={"Authorization": "Bearer " + bearer_token,
             "Accepth-Encoding": "gzip, deflate, br"}
    url = endpoint + "motors"
    r = requests.get(url=url, headers=headers)

    print(r)

    motors_file = open("documentation")
    json.dump(r.json())


def get_specific_motor(bearer_token, motor_id):
    # https: // your - api - endpoint.otosensesms.com / motors
    headers={"Authorization": "Bearer " + bearer_token,
             "Accepth-Encoding": "gzip, deflate, br"}
    url = endpoint + "motors" + "/" + motor_id
    r = requests.get(url=url, headers=headers)

    print(r)
    print(r.json())


def get_data(bearer_token, motor_id, dataset):
    # Seems to be up to 120 sets of records for 1 request
    # But that might be different depending on the dataset

    # https: // your - api - endpoint.otosensesms.com / data / {motorId} / {dataset}

    headers = {"Authorization": "Bearer " + bearer_token,
               "Accepth-Encoding": "gzip, deflate, br"}

    datasets = ["vibx", "vibz", "flux", "tempe", "tempm", "performance", "conditions", "operations", "vibxFFT", "vibzFFT", "fluxFFT"]
    url = endpoint + "data/" + motor_id + "/" + datasets[0]

    print(url)

    con_token = ""

    # ISO-8601 date and time
    '2022-03-01T11:23:19.715Z'
    start = "2021-01-01T00:00:00Z"
    end = "2022-02-06T13:38:14Z"
    payload = {'start': start, 'end': end}

    r = requests.get(url=url, headers=headers, params=payload)

    print(r)

    res_file = open("documentation/results2.json", "w")
    json.dump(r.json(), res_file)


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
    bearer_token = read_bearer_token()

    print(bearer_token)

    get_all_motors(bearer_token)

    print("---------MOTOR----------")
    # get_specific_motor(bearer_token, motor_id3)

    # get_data(bearer_token, motor_id3)

