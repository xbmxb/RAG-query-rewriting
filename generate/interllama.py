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
import torch
# from llama import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig


def add_prompt(item, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question'] # only for chatbot
    prompt = prompt.replace('{query}', query)
    if item.get('passage'):
        prompt = prompt.replace('{passage}', item['passage'])
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
        prompt = prompt.replace('{background}', backinfo)
        prompt = prompt.replace('{output}', backinfo)
    return prompt

def add_prompt_llama(item, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question'] # only for chatbot
    prompt = prompt.replace('{query}', query)
    if item.get('passage'):
        prompt = prompt.replace('{passage}', item['passage'])
    maxlength = 1800
    if "output" in item.keys(): # background info
        
        if type(item['output']) == list:
            backinfo = rmreturn(item['output'][0])
        elif type(item['output']) == str:
             backinfo = rmreturn(item['output'])
        toker = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = toker.encode(backinfo)
        print("original length: ", len(tokens))
        if len(tokens) > maxlength:
            print("trunc!! original length: ", len(tokens))
            tokens = tokens[:maxlength]
        backinfo = toker.decode(tokens)
        prompt = prompt.replace('{background}', backinfo)
        prompt = prompt.replace('{output}', backinfo)
    return prompt
 
def completion_llama(model, device, tokenizer, prompt, max_tokens=100, temperature=0, logprobs=None, n=1,
    frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False,
    partition_id=None, **kwargs):

    generation_config = dict(
        temperature=temperature,
        num_beams=1,
        repetition_penalty=1.3,
        max_new_tokens=max_tokens
        )
    with torch.no_grad():
        print("Start inference.")
        results = []
        for index, example in enumerate(prompt):
            input_text = example
            inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device), 
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **generation_config
            )
            s = generation_output[0]
            output = tokenizer.decode(s,skip_special_tokens=True)
            # if "### Response:" in output:
            print('total output: ', output)
            output = output.split("Answer:")[-1].strip()
            # print(output, stop)
            output = output.split('\n')[0].strip()
            # output = output.split(stop)[0].strip()
            print(f"======={index}=======")
            print(f"Input: {example}\n")
            print(f"Output: {output}\n")

            results.append(output)
    return results

def run_main(inlines, outfile, engine, prompt, max_tokens, n=1, temp=0, end=None):

    config = LlamaConfig.from_pretrained(engine)
    tokenizer = LlamaTokenizer.from_pretrained(engine)
    print('loading llm: ', engine)
    starttime = time.time()
    llama = LlamaForCausalLM.from_pretrained(
        engine,
        torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    endtime = time.time()
    print('llm loaded: '+ str(endtime-starttime))
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    print("using device: ", device)
    batch = 20
    llama.to(device)

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
        for _ in range(batch):
            if index >= len(inlines): break
            input_with_prompt = add_prompt_llama(inlines[index], prompt)
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
        # outputs = complete(inputs_with_prompts, max_tokens=max_tokens, temperature=temp, n=n, stop=end)
        outputs = completion_llama(llama, device, tokenizer, prompt=inputs_with_prompts, max_tokens=max_tokens, temperature=temp, stop=end)
        for j, output in enumerate(outputs):
            # output = output.split(prompt)[1]
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