#Copyright (c) Microsoft Corporation. All rights reserved.
#Licensed under the MIT License.

# -*- coding: utf-8 -*-

import json, time, logging
import os 
from pprint import pprint
import requests

def searchbing(query):
    # Add your Bing Search V7 subscription key and endpoint to your environment variables.
    subscription_key = ''
    endpoint = ''

    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    # Call the API
    retry_interval_exp = 0
    while True:
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as ex:
            logging.warning("Exception...")
            if retry_interval_exp > 6:
                return {}
            time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1


def searchbing_filter(query):
    # Add your Bing Search V7 subscription key and endpoint to your environment variables.
    subscription_key = ''
    endpoint = ''

    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    # Call the API
    retry_interval_exp = 0
    while True:
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as ex:
            logging.warning("Exception...")
            if retry_interval_exp > 6:
                return {}
            time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1