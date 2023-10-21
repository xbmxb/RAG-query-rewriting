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

 
# completion function of ChatGPT API
def complete(
    prompt, max_tokens=100, temperature=0, logprobs=None, n=1,
    frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False,
    partition_id=None, **kwargs
) -> str:
    pass

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