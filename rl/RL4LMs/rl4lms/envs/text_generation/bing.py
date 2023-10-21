#Copyright (c) Microsoft Corporation. All rights reserved.
#Licensed under the MIT License.

# -*- coding: utf-8 -*-

import json, time, logging
import os 
from pprint import pprint
import requests

def searchbing(query):
    # Add your Bing Search V7 subscription key and endpoint to your environment variables.
    subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
    endpoint = os.environ['BING_SEARCH_V7_ENDPOINT'] + "v7.0/search"

    # Query term(s) to search for. 
    # query = "Harry Potter"

    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    # Call the API
    retry_interval_exp = 0
    while True:
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            # print("\nHeaders:\n")
            # print(response.headers)

            # print("\nJSON Response:\n")
            # pprint(response.json())
            return response.json()
        except Exception as ex:
            logging.warning("Exception...")
            if retry_interval_exp > 20:
                raise ex
            time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1