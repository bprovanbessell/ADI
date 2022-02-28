"""https://adi.otosensesms.com/api-reference

"""

import requests

adi_url = ""

# Add to path (url?)
# https://your-api-endpoint.otosensesms.com/data/{motorId}/{dataset}


headers = {
    'accept-encoding': 'gzip, deflate, br'
}

# continuation token needed in the params
params = ""

# auth

# accept encoding
r = requests.get(adi_url, headers="", )

# s = requests.Session()
# s.auth = ('user', 'pass')
# s.headers.update({'x-test': 'true'})
