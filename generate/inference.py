import openai
import os
import time
import threading
import json
import _thread
from tqdm import tqdm
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager
from collections import defaultdict
from transformers import GPT2Tokenizer

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def add_prompt(item, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question'] # only for chatbot
    # query = item['input']
    # if item.get('thought'):
    #     query = query + ' Thought: ' + item['thought']
    prompt = prompt.replace('{query}', query)
    if item.get('passage'):
        prompt = prompt.replace('{passage}', item['passage'])
    # prompt = prompt.replace('{answer}', item['answer'][0]) # for filter

    # This model's maximum context length is 8193 tokens -> 7200
    maxlength = 3200
    if "output" in item.keys(): # background info
        if type(item['output']) == list:
            backinfo = rmreturn(item['output'][0])
        elif type(item['output']) == str:
             backinfo = rmreturn(item['output'])
        toker = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = toker.encode(backinfo)
        if len(tokens) > maxlength:
            tokens = tokens[:maxlength]
        backinfo = toker.decode(tokens)
        # backinfo_trunc = maxlength - 5 * len(prompt.split(' '))
        # # print(backinfo_trunc)
        # # print(len(backinfo.split(' ')))
        # if len(backinfo.split(' ')) > backinfo_trunc:
        #     # print(len(backinfo.split(' ')))
        #     backinfo = ' '.join(backinfo.split(' ')[:backinfo_trunc])
        #     # print(len(backinfo.split(' ')))
        # # print("backinfo: ", backinfo)
        prompt = prompt.replace('{background}', backinfo)
        prompt = prompt.replace('{output}', backinfo)

    return prompt


def clustering_prompt(items, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    cluster_prompts = []
    for item in items:
        query = item['question']
        backinfo = rmreturn(item['output'][0])
        item_prompt = prompt.replace('{query}', query)
        item_prompt += f' {backinfo}'
        cluster_prompts.append(item_prompt)

    cluster_prompts.append(prompt)
    return ' \n\n\n\n '.join(cluster_prompts)


def run_embeddings(input_text, engine='text-similarity-davinci-001'):
    
    texts = [t.replace('\n', '') for t in input_text]
    outputs = openai.Embedding.create(input=texts, model=engine)['data']
    embeddings = [o['embedding'] for o in outputs]

    return embeddings


def run_inference(inputs_with_prompts, engine, max_tokens, num_sequence=1, temp=0):

    completions = {"choices": []}
    for _ in range(200):
        try:
            with time_limit(20, 'run gpt-3'):
                completions = openai.Completion.create(
                    engine=engine, 
                    max_tokens=max_tokens, 
                    prompt=inputs_with_prompts, 
                    temperature=temp, 
                    n=num_sequence, # num of returned sequence
                    )
                break
        except:
                time.sleep(2)

    outputs = [c["text"] for c in completions["choices"]]
    return outputs


 
# class ChatDavinci002:
def complete(
    prompt, max_tokens=100, temperature=0, logprobs=None, n=1,
    frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False,
    partition_id=None, **kwargs
) -> str:
    openai.api_base = "https://xsum-eus.openai.azure.com/"
    api_key = os.environ.get('OPENAI_ACCESS_KEY', None)
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"
    # deployment_id = "chatgpt"
    deployment_id = 'gpt35' # davinci003
    
    # api_key = "b1d842155ce6400096b013d810d6ab2b"
    # deployment_id = "gpt-35-turbo"
    # openai.api_base = "https://tscience-aoai-eus.openai.azure.com/" 
    # openai.api_type = "azure"
    # openai.api_version = "2023-03-15-preview"
    openai.api_key = api_key
    if rstrip:
        # Remove heading whitespaces improves generation stability. It is
        # disabled by default to keep consistency.
        prompt = prompt.rstrip()
    retry_interval_exp = 1 
    # temperature=0.36 # onlyl for chatbot
    while True:
        try:
            # Partition ID should only be set explictly when requests can
            # utilize server side cache. Caching helps reduce computational
            # cost for prompts with similar content. You may manually
            # assign the same partition ID to similar prompts.
            if not partition_id:
                # Requests will be routed in round robin by default.
                partition_id = f"sumscience-{datetime.now()}"
            completions = openai.Completion.create(
                deployment_id=deployment_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,  # Not recommended to change with temperature
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                # logprobs=logprobs,
                n=n,
                stop=stop,
                headers={"partition-id": partition_id},
                **kwargs,
            )
            if 'choices' not in completions.keys():
                #error:
                completions['choices'] = [{'text': 'Not found.'}]
            outputs = [c["text"] for c in completions["choices"]]
            return outputs
        except openai.error.RateLimitError as e:
            # NOTE: openai.error.RateLimitError: Requests to the
            # Deployments_Completion Operation under OpenAI API have
            # exceeded rate limit of your current OpenAI S0 pricing tier.
            # Please retry after 7 seconds. Please contact Azure support
            # service if you would like to further increase the default rate
            # limit.
            print(e)
            logging.warning("OpenAI rate limit error. Retry")
            # Expontial backoff
            time.sleep(min(10, 0.5 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1
        except openai.error.InvalidRequestError as e:
            print(e)
            return ''

def run_main(inlines, outfile, engine, prompt, max_tokens, n=1, temp=0, end=None):

    if os.path.exists(outfile):
        outs = open(outfile, 'a', encoding='utf8')
        num_lines = len(open(outfile, 'r').readlines())
        inlines = inlines[num_lines - 1: ]
    else: # not os.path.exists(outfile)
        outs = open(outfile, 'a', encoding='utf8')
        outs.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + '\n')
    # inlines = inlines[:2]
    # print(inlines)
    pbar = tqdm(total = len(inlines))
    index = 0
    pbar.update(index)
    while index < len(inlines):
        inputs, answers = [], []
        inputs_with_prompts = []
        for _ in range(20):
            if index >= len(inlines): break
            input_with_prompt = add_prompt(inlines[index], prompt)
            if index == 0: 
                print(input_with_prompt)
            # os._exit()
            inputs.append(inlines[index]['question']) ## a string
            answers.append(inlines[index]['answer']) ## a list of strings only for chatbot
            # inputs.append(inlines[index]['input']) ## a string
            # answers.append(inlines[index]['output']) 
            inputs_with_prompts.append(input_with_prompt)
            index += 1
        # print(inputs_with_prompts)
        samples = defaultdict(list)
        outputs = complete(inputs_with_prompts, max_tokens=max_tokens, temperature=temp, n=n, stop=end)
        for j, output in enumerate(outputs):
            samples[j//n].append(output)

        for i in range(len(inputs_with_prompts)):
            outs.write(json.dumps({
                'question': inputs[i], 
                'answer': answers[i], 
                'output': samples[i]}, ensure_ascii=False) 
                +'\n')

        pbar.update(len(inputs_with_prompts))

    pbar.close()
    outs.close()

def run_searchre(inlines, outfile, engine, prompt, max_tokens, n=1, temp=0, end=None):

    if os.path.exists(outfile):
        outs = open(outfile, 'a', encoding='utf8')
        num_lines = len(open(outfile, 'r').readlines())
        inlines = inlines[num_lines - 1: ]
    else: # not os.path.exists(outfile)
        outs = open(outfile, 'a', encoding='utf8')
        outs.write(json.dumps({"prompt": prompt}) + '\n')
    # inlines = inlines[:2]
    # print(inlines)
    pbar = tqdm(total = len(inlines))
    index = 0
    pbar.update(index)
    while index < len(inlines):
        inputs, answers = [], []
        inputs_with_prompts = []
        for _ in range(20):
            if index >= len(inlines): break
            input_with_prompt = add_prompt(inlines[index], prompt)
            if index == 0: 
                print(input_with_prompt)
            # os._exit()
            inputs.append(inlines[index]['question']) ## a string
            # answers.append(inlines[index]['answer']) ## a list of strings
            inputs_with_prompts.append(input_with_prompt)
            index += 1
        # print(inputs_with_prompts)
        samples = defaultdict(list)
        outputs = complete(inputs_with_prompts, max_tokens=max_tokens, temperature=temp, n=n, stop=end)
        for j, output in enumerate(outputs):
            samples[j//n].append(output)

        for i in range(len(inputs_with_prompts)):
            outs.write(json.dumps({
                'question': inputs[i], 
                # 'answer': answers[i], 
                'output': samples[i]}) 
                +'\n')

        pbar.update(len(inputs_with_prompts))

    pbar.close()
    outs.close()
    return 

def run_main_search(inlines, outfile, engine, prompt, max_tokens, n=1, temp=0, end=None):

    if os.path.exists(outfile):
        outs = open(outfile, 'a', encoding='utf8')
        num_lines = len(open(outfile, 'r').readlines())
        inlines = inlines[num_lines - 1: ]
    else: # not os.path.exists(outfile)
        outs = open(outfile, 'a', encoding='utf8')
        outs.write(json.dumps({"prompt": prompt}) + '\n')
    # inlines = inlines[:2]
    # print(inlines)
    pbar = tqdm(total = len(inlines))
    index = 0
    pbar.update(index)
    while index < len(inlines):
        inputs, answers = [], []
        inputs_with_prompts = []
        for _ in range(20):
            if index >= len(inlines): break
            input_with_prompt = add_prompt(inlines[index], prompt)
            inputs.append(inlines[index]['question']) ## a string
            answers.append(inlines[index]['answer']) ## a list of strings
            inputs_with_prompts.append(input_with_prompt)
            index += 1
        # print(inputs_with_prompts)
        samples = defaultdict(list)
        outputs = complete(inputs_with_prompts, max_tokens=max_tokens, temperature=temp, n=n, stop=end)
        for j, output in enumerate(outputs):
            samples[j//n].append(output)

        for i in range(len(inputs_with_prompts)):
            outs.write(json.dumps({
                'question': inputs[i], 
                'answer': answers[i], 
                'output': samples[i]}) 
                +'\n')

        pbar.update(len(inputs_with_prompts))

    pbar.close()
    outs.close()