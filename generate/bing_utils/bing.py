#Copyright (c) Microsoft Corporation. All rights reserved.
#Licensed under the MIT License.

# -*- coding: utf-8 -*-

import json, logging, time
import os 
from pprint import pprint
import requests
from bs4 import BeautifulSoup
from .bm25skl import bm25score
import regex, string

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

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

def morer(r, itnn):
    # pprint(r)
    if 'rankingResponse' not in r.keys():
        return []
    if 'mainline' not in r['rankingResponse'].keys():
        return []
    ranking = r['rankingResponse']['mainline']['items']
    itn = 0
    urls = []
    snippets = []
    for it in ranking:
        # if itn >= itnn:
        #     break
        # print(it)
        if 'value' not in it.keys():
            continue
        atype = it['answerType'][0].lower() + it['answerType'][1:]
        # only text
        if atype == 'webPages':
            for it_ in r[atype]['value']:
                # print(it_)
                if it_['id'] == it['value']['id']:
                    # itn += 1
                    urls.append(it_['url'])
                    snippets.append(it_['snippet'])
    # step in
    docss = []
    for i, ui in enumerate(urls):
        if itn >= itnn:
            break
        print(i, 'STEP IN: ', ui)
        retry_interval_exp = 1
        old_time = time.time()
        found = False
        while retry_interval_exp < 3 and not found:
            try:
                start_time = time.time()
                response_text = requests.get(ui, timeout=6)
                found = True
            except requests.exceptions.ConnectionError:
                time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                print("requests.exceptions.ConnectionError, retry: ", retry_interval_exp)
                retry_interval_exp += 1
            except requests.exceptions.ReadTimeout:
                print("requests.exceptions.ReadTimeout.")
                break
            except requests.exceptions.ContentDecodingError:
                print("requests.exceptions.ContentDecodingError.")
                break
            except requests.exceptions.TooManyRedirects:
                print('requests.exceptions.TooManyRedirects')
                break
            except requests.exceptions.ChunkedEncodingError:
                print('requests.exceptions.ChunkedEncodingError')
                break
        if not found:
            doci = []
            doci.append(snippets[i])
        else:
            response_text = response_text.text
            # print(response_text)
            search_time = time.time() - old_time
            try:
                soup = BeautifulSoup(response_text, features="html.parser")
                # print('======================')
                # print(soup)
                ptext = soup.find_all('p')
            except:
                ptext = []
            # if ptext == []:
            #     continue
            doci = []
            doci.append(snippets[i])
            for ptexti in ptext:
                doci.append(ptexti.get_text())

            if doci == [] or sum([len(i.strip()) for i in doci]) == 0:
                continue

        itn += 1
        # print('======================')
        # print(ptext)
        # print('======================')
        # print(doci)
        # doci = ' '.join(doci)
        # print('======================')
        # print(len(doci), doci)
        docss.append(doci)
    return docss
    # select part of this doc
    # BM25
    # doc = bm25score(docs=doci, q=query, max_words=1000)
    # return doc
    
# print('test bing ')
# query = 'Which name is given to the heart chamber which receives blood?'
# r = searchbing(query)
# print("\nJSON Response:\n")
# pprint(r)
# docss = morer(r, 1)
# # bm25
# search_res = []
# for docs in docss:
#     print('docs: ', docs)
#     doc = bm25score(docs=docs, q=query, max_words=1000)
#     search_res.append(doc)
# print(search_res)

def searchsele(query, topn, max_words_perdoc):

    r = searchbing(query)
    # print("\nJSON Response:\n")
    # pprint(r)
    docss = morer(r, topn)
    if docss == []:
        print('find nothing: ', r)
        return ''
    # bm25
    search_res = []
    print('bm25: ', query)
    for doci in docss:
        # print('docs: ', doci)
        doc = bm25score(docs=doci, q=query, max_words=max_words_perdoc, topp=0.2, use='words')
        search_res.append(doc)
    # print(search_res)
    return search_res

def searchbl(query, topn, gold):
    r = searchbing(query)
    # print("\nJSON Response:\n")
    # pprint(r)
    docss = morer(r, topn)
    if docss == []:
        print('find nothing: ', r)
        return ''
    search_res = []
    for doci in docss:
        hit = [1 if normalize_answer(i) in normalize_answer(" ".join(doci)) else 0 for i in gold]
        if sum(hit) > 0:
            search_res = doci
            break
    return search_res

def searchrdoc(query, topn):
    r = searchbing(query)
    docss = morer(r, topn)
    if docss == []:
        print('find nothing: ', r)
        return ''
    # assert len(docss) == 1
    return " ".join(docss)

def searchr1(query, topn):
    r = searchbing(query)
    docss = morer(r, topn)
    if docss == []:
        print('find nothing: ', r)
        return ''
    # assert len(docss) == 1
    return " ".join(docss)