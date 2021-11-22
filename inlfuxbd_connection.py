from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# You can generate a Token from the "Tokens Tab" in the UI
token = "PExKkk9IIRJSS9TJQbUv0aoNZa8v2DbpUFyJXSzzmAiudxrrmHj6B_uzW6tawZiKkSS7U87TfOvZeyuYJsLpSg=="
org = "UCC"
bucket = "SmartSensor"

client = InfluxDBClient(url="http://localhost:8086", token=token)

write_api = client.write_api(write_options=SYNCHRONOUS)

# data = "mem,host=host1 used_percent=23.43234543"
p = Point("sample").tag("signal", "current").tag("machine", "PU7001").field("data", 1.25)
# p = Point("sample").tag("signal", "flux").tag("machine", "PU7001").field("data", 1.25)
write_api.write(bucket=bucket, org=org, record=p)

# query_api = client.query_api()
#
# query = 'from(bucket:"Sma rtSensorADI")|> range(start: 3)'
# result = client.query_api().query(org=org, query=query)

# results = []
# for table in result:
#     for record in table.records:
#         results.append((record.get_value(), record.get_field()))
#
# print(results)