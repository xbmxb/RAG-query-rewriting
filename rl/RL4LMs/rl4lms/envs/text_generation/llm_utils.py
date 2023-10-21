import openai
import jsonlines,logging,os,json,time
from tqdm import tqdm
from .wikienv import WikiEnv
from .bingenv import BingEnv
from collections import defaultdict
from datetime import datetime, timedelta
from transformers import GPT2Tokenizer

# rewrite2con prompt
PROMPT = [
    "Answer the question with just one entity in the following format, end the answer with '**'. \n\n Question: World heavyweight champion. Charles returned to boxing after the war as a light heavyweight, picking up many notable wins over leading light heavyweights, as well as heavyweight contenders Archie Moore, Jimmy Bivins, Lloyd Marshall and Elmer Ray. Ezzard Charles was a world champion in which sport? \n\n Answer: Prize fight** \n\n Question: Nitrous oxide (dinitrogen oxide or dinitrogen monoxide), commonly known as laughing gas, nitrous, or nos, is a chemical compound, an oxide of nitrogen with the formula N2O. At room temperature, it is a colourless non-flammable gas, and has a slightly sweet scent and taste.[5] At elevated temperatures, nitrous oxide is a powerful oxidiser similar to molecular oxygen. What is the correct name of laughing gas? \n\n Answer: Nitrous oxide**  \n\n Question: The right and left atria are the top chambers of the heart and receive blood into the heart. The right atrium receives deoxygenated blood from systemic circulation and the left atrium receives oxygenated blood from pulmonary circulation. The atria do not have inlet valves, but are separated from the ventricles by valves. Which name is given to the heart chamber which receives blood? \n\n Answer: Atrium** \n\n Question: {background} {query} \n\n Answer: ",
    "Answer the question with just one entity, not sentences, end the answer with '**'. \n\n Question: Darwin's Nightmare is a 2004 Austrian-French-Belgian documentary film written and directed by Hubert Sauper, dealing with the environmental and social effects of the fishing industry around Lake Victoria in Tanzania. It premiered at the 2004 Venice Film Festival, and was nominated for the 2006 Academy Award for Best Documentary Feature at the 78th Academy Awards.[1] The Mouth of the Wolf (original title: La bocca del lupo) is a 2009 biographical drama/documentary film written and directed by Pietro Marcello. It premièred at the 2009 Torino Film Festival in Turin, and won the FIPRESCI Prize for 'Best Film'[1] and the Prize of the City of Torino. In 2010 it appeared at the 60th Berlin International Film Festival where it won the Caligari Film Award and the Teddy Award for 'Best Documentary' Which documentary premiered at a film festival first, Darwin's Nightmare or The Mouth of the Wolf? \n Answer: Darwin's Nightmare** \n\nQuestion: '''First sighted by the Dutch explorer Abel Tasman, the country was later mapped by James Cook, the British seafarer who dominates the story of the European discovery of New Zealand. which highway is named after a seafarer who discovered New Zealand?''' \nAnswer: Captain Cook Highway** \n\nQuestion: '''{background} {query}''' \nAnswer: ",
    "按照如下格式回答下列问题，选出一个或多个正确选项，回答请以‘**’作为结尾。 \n\n 你是否希望你居住的地方既能享受乡村的绿树环绕又享受城市的便利？在不久的将来，也许会成为一种可能。中国第一个“垂直森林城市”已经建成，2022 年 1 月 - 黄冈居然之家垂直森林城市综合体已经迎来了它的第一批住户。这座垂直森林为大约 500 人提供了一个家 - 以及超过 5,000 株灌木和树木。居然之家占地4.54公顷，包括两座住宅和三座商业塔。住宅塔楼结合了开放式和封闭式阳台，形成了楼梯般的效果。这种设计旨在“创造一个连续的、不断变化的动态空间”，塔楼外立面采用悬挑元素，结合阳台空间打破建筑的规整性，也为植物的生长提供了一个富有层次和变化的平台，阳台延伸到外部空间，让久居城市的人们也能充分感受被自然环绕的惬意。问题：在某城市中心，一种创新型绿色建筑一垂直森林高层住宅落成面世。它是在建筑的垂直方向上，覆盖满本地乔木、灌木和草本等植物，为每层住户营造“空中花园”，形成具有森林效应的生态居住群落。与传统设计相比，“垂直森林”在居住空间设计上变化最大的地方是（）选项: (A)阳台,(B)客厅,(C)卧室,(D)厨房 \n\n回答：A** \n\n {background} 问题: {query} \n\n 回答:",
    "Answer the question with just one entity in the following format, end the answer with '**'. \n\n Question: World heavyweight champion. Charles returned to boxing after the war as a light heavyweight, picking up many notable wins over leading light heavyweights, as well as heavyweight contenders Archie Moore, Jimmy Bivins, Lloyd Marshall and Elmer Ray. Ezzard Charles was a world champion in which sport? \n\n Answer: Prize fight.** \n\n Question: Nitrous oxide (dinitrogen oxide or dinitrogen monoxide), commonly known as laughing gas, nitrous, or nos, is a chemical compound, an oxide of nitrogen with the formula N2O. At room temperature, it is a colourless non-flammable gas, and has a slightly sweet scent and taste.[5] At elevated temperatures, nitrous oxide is a powerful oxidiser similar to molecular oxygen. What is the correct name of laughing gas? \n\n Answer: Nitrous oxide.** \n\n Question: {background} {query} \n\n Answer: "
]

# def add_prompt(item, prompt):
#     def rmreturn(s):
#         s = s.replace('\n\n', ' ')
#         s = s.replace('\n', ' ')
#         return s.strip()

#     query = item['question']
#     prompt = prompt.replace('{query}', query)

#     if item.get('output'): # background info
#         if type(item['output']) == list:
#             backinfo = rmreturn(item['output'][0])
#         elif type(item['output']) == str:
#             backinfo = rmreturn(item['output'])
#         prompt = prompt.replace('{background}', backinfo)

#     return prompt

def add_prompt(item, prompt):
    toker = GPT2Tokenizer.from_pretrained('gpt2')
    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question'] # only for chatbot
    prompt = prompt.replace('{query}', query)
    # if item.get('passage'):
    #     prompt = prompt.replace('{passage}', item['passage'])
    current_len = len(toker.encode(prompt))
    # maxlength = 7200
    totallen = 7200
    maxlength = totallen - current_len
    # print(maxlength)
    if "output" in item.keys(): # background info
        if type(item['output']) == list:
            backinfo = rmreturn(item['output'][0])
        elif type(item['output']) == str:
             backinfo = rmreturn(item['output'])
        # toker = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = toker.encode(backinfo)
        print('output len tokens: ', len(tokens), ' but max len: ', maxlength)
        if len(tokens) > maxlength:
            tokens = tokens[:maxlength]
            print('trunc to: ', len(tokens))
        backinfo = toker.decode(tokens)
        prompt = prompt.replace('{background}', backinfo)
        totaltokens = toker.encode(prompt)
        assert totallen >= len(totaltokens)
    return prompt

def complete(
    prompt, max_tokens=100, temperature=0, logprobs=None, n=1,
    frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False,
    partition_id=None, **kwargs
) -> str:
    # print("stop by ", stop)
    openai.api_base = "https://xsum-eus.openai.azure.com/"
    openai.api_type = "azure"
    # openai.api_version = "2022-12-01"
    openai.api_version = "2023-03-15-preview" 
    deployment_id = "chatgpt"
    api_key = os.environ.get('OPENAI_ACCESS_KEY', None)
    if rstrip:
        prompt = prompt.rstrip()
    retry_interval_exp = 1 
    while True:
        try:
            if not partition_id:
                partition_id = f"sumscience-{datetime.now()}"
            completions = openai.Completion.create(
                engine=deployment_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,  # Not recommended to change with temperature
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
            print(e)
            logging.warning("OpenAI rate limit error. Retry "+ str(min(10, 0.5 * (2 ** retry_interval_exp))))
            # Expontial backoff
            time.sleep(min(10, 0.5 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1
            if retry_interval_exp >= 50:
                return 'InvalidRequestError'
        except openai.error.APIConnectionError as e:
            print(e)
            logging.warning("OpenAI APIConnectionError. Retry "+ str(min(20, 2 * (2 ** retry_interval_exp))))
            # Expontial backoff
            time.sleep(min(10, 2 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1
        except openai.error.InvalidRequestError as e:
            print(e)
            logging.warning("openai.error.InvalidRequestError. Return nothing.")
            logging.warning(prompt)
            return 'InvalidRequestError'
        except openai.error.Timeout as e:
            print(e)
            logging.warning("timeout. Retry "+ str(min(20, 2 * (2 ** retry_interval_exp))))
            # Expontial backoff
            time.sleep(min(20, 2 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1

def complete_aoai(
    prompt, origq, max_tokens=100, temperature=0, logprobs=None, n=1,
    frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False,
    partition_id=None, **kwargs
) -> str:
    # print("stop by ", stop)
    # openai.api_base = "https://xsum-eus.openai.azure.com/"
    # openai.api_type = "azure"
    # openai.api_version = "2022-12-01"
    # deployment_id = "chatgpt"
    # api_key = os.environ.get('OPENAI_ACCESS_KEY', None)
    deployment_id = "gpt-35-turbo"
    api_key = "b1d842155ce6400096b013d810d6ab2b"
    openai.api_type = "azure" 
    openai.api_base = "https://tscience-aoai-eus.openai.azure.com/" 
    openai.api_version = "2022-12-01" 
    openai.api_key = api_key
    if rstrip:
        prompt = prompt.rstrip()
    retry_interval_exp = 1 
    while True:
        try:
            if not partition_id:
                partition_id = f"sumscience-{datetime.now()}"
            completions = openai.Completion.create(
                engine=deployment_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,  # Not recommended to change with temperature
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
                completions['choices'] = [{'text': ' '}]
            outputs = [c["text"] for c in completions["choices"]]
            return outputs

        except openai.error.RateLimitError as e:
            logging.warning("OpenAI rate limit error. Retry")
            # Expontial backoff
            time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1
        except openai.error.APIConnectionError as e:
            logging.warning("OpenAI APIConnectionError. Retry")
            # Expontial backoff
            time.sleep(max(20, 2 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1
        except openai.error.InvalidRequestError as e:
            logging.warning("openai.error.InvalidRequestError. Return nothing.")
            logging.warning(prompt)
            try:
                if not partition_id:
                    partition_id = f"sumscience-{datetime.now()}"
                print("invalid search result? try original question: ", origq)
                completions = openai.Completion.create(
                    engine=deployment_id,
                    prompt=origq,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,  # Not recommended to change with temperature
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
                    completions['choices'] = [{'text': ' '}]
                outputs = [c["text"] for c in completions["choices"]]
                return outputs
            except: 
                return 'InvalidRequestError'
        except openai.error.Timeout as e:
            logging.warning("timeout. Retry")
            # Expontial backoff
            time.sleep(max(60, 2 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1

def run_main(inlinesllm, outputfile, prompt, max_tokens, n, temp, bar, end="**"):
    # print(bar)
    inlines = inlinesllm
    outs = open(outputfile,'a',encoding='utf8')
    if bar:
        pbar = tqdm(total = len(inlines))
    index = 0
    if bar:
        pbar.update(index)
    return_ans = []
    if len(inlines) > 1:
        print(f"llm got {str(len(inlines))} samples.")
    while index < len(inlines):
        if index % 50 == 0 and index != 0:
            print(f"processing {str(index)}-th sample with llm...")
        inputs, answers = [], []
        inputs_with_prompts = []
        # origqs = []
        for _ in range(20):
            if index >= len(inlines): break
            input_with_prompt = add_prompt(inlines[index], prompt)
            # print(input_with_prompt)
            inputs.append(inlines[index]['question']) ## a string
            # answers.append(inlines[index]['answer']) ## a list of strings
            inputs_with_prompts.append(input_with_prompt)
            # orig_prompt = add_prompt({'question':inlines[index]['question']}, PROMPT[4])
            # origqs.append(orig_prompt)
            index += 1
        # print(inputs_with_prompts)
        samples = defaultdict(list)
        # outputs = complete_aoai(inputs_with_prompts, origqs, max_tokens=max_tokens, temperature=temp, n=n, stop=end)
        outputs = complete(inputs_with_prompts, max_tokens=max_tokens, temperature=temp, n=n, stop=end)
        for j, output in enumerate(outputs):
            samples[j//n].append(output)

        for i in range(len(inputs_with_prompts)):
            outs.write(json.dumps({
                'question': inputs[i], 
                'prompt': inputs_with_prompts[i],
                # 'answer': answers[i], 
                'output': samples[i]}) 
                +'\n')
            return_ans.append(samples[i])
        if bar:
            pbar.update(len(inputs_with_prompts))
    if bar:
        pbar.close()
    outs.close()
    return return_ans

def llm(questions, queries, pid, bar, searchfunc = 'plain', topn=None, max_words_perdoc=None, think = False, max_obs = None, n=1, temp=0, end="**"):
    # search in wikienv 
    sele = False
    gold = None
    # think = True
    max_tokens = 48
    prompt = PROMPT[pid]
    outputfile = './outputs_of_llm.jsonl'
    bingenv = BingEnv()
    # wikienv.appendsimilar = True
    nowiki = 0
    inlinesllm = []
    if bar:
        for il in tqdm(range(len(questions))):
            ilm = {
                "question": questions[il],
            }
            if think:
                qs = queries[il]
                if 'Query' in qs:
                    qs = [qs.split("Query")[1]]
                elif 'query' in qs:
                    qs = [qs.split("query")[1]]
                if ";" in qs:
                    qs = qs.split(";")
                else:
                    qs = [queries[il]]
            else:
                qs = [queries[il]]
            # print(queries[il])
            # print(qs)
            ilm['output'] = []
            for q in qs:
                query = 'search[' + q +']'
                obs, reward, done, info = bingenv.step(query, False, searchfunc, gold, topn=None, max_words_perdoc=None)
                if sele == True and obs.startswith("Similar:"):
                    ilm['output'].append([''])
                    nowiki += 1
                else:
                    if max_obs:
                        obs = " ".join(obs.split()[:max_obs])
                    ilm['output'].append(obs)
            ilm['output'] = ' '.join(ilm['output'])
            inlinesllm.append(ilm)
            # print(obs)
            # os._exit()
    else:
        for il in range(len(questions)):
            ilm = {
                "question": questions[il],
            }
            # print("think: ", think)
            # print("queries: ", queries[il])
            if think:
                qs = queries[il]
                if 'Query' in qs:
                    qs = [qs.split("Query")[1]]
                elif 'query' in qs:
                    qs = [qs.split("query")[1]]
                if ";" in qs:
                    qs = qs.split(";")
            else:
                qs = [queries[il]]
            print(qs,questions)
            ilm['output'] = []
            for q in qs:
                query = 'search[' + q +']'
                obs, reward, done, info = bingenv.step(query, False, searchfunc, gold, topn=None, max_words_perdoc=None)
                if sele == True and obs.startswith("Similar:"):
                    ilm['output'].append([''])
                    nowiki += 1
                else:
                    if max_obs:
                        obs = " ".join(obs.split()[:max_obs])
                    ilm['output'].append(obs)
            ilm['output'] = ' '.join(ilm['output'])
            inlinesllm.append(ilm)
            # print(obs)
            # os._exit()
    samples = run_main(inlinesllm, outputfile, prompt, max_tokens, n, temp, bar, end)
    return samples, inlinesllm