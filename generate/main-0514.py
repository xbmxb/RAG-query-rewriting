import argparse
import os
import json

from inference import run_main, run_searchre
from evaluation import (
    eval_recall,
    eval_question_answering,
    eval_fact_checking,
    eval_dialogue_system,
    hits,f1, ems
)
import wikienv
from bingenv import BingEnv
from tqdm import tqdm

def datapath(dataset, split):
    if dataset == 'tqa-eg':
        inputfile = '/xinbei_data/replug/tqa/egs/tqa-test-eg.jsonl'
    elif dataset == 'tqa':
        inputfile = f'/xinbei_data/replug/tqa/tqa-{split}.jsonl'
    elif dataset == 'hotpot':
        inputfile = f'/xinbei_data/replug/ReAct/data/hotpot_{split}_v1_simplified.json'
    elif dataset == 'nq':
        inputfile = f'/xinbei_data/replug/nq/nq-{split}.jsonl'
    elif dataset == 'tqa-wrong':
        inputfile = f'/xinbei_data/replug/generate/forRL/rewrite-tqa/wrongcases.jsonl'
    elif dataset == 'tqa-totalwrong':
        inputfile = f'/xinbei_data/replug/generate/forRL/rewrite-tqa/totalwrongcases.jsonl'
    elif dataset == 'tqa-filter':
        inputfile = f'/xinbei_data/replug/generate/filtered-nq/tqa-{split}-filtered.jsonl'
    elif dataset == 'nq-filter':
        inputfile = f'/xinbei_data/replug/generate/filtered-nq/nq-{split}-filtered.jsonl' 
    elif dataset == 'webq':
        inputfile = f'/xinbei_data/replug/webq/webq-{split}.jsonl'
    elif dataset == 'amb':
        inputfile = f'/xinbei_data/replug/ambignq/{split}.jsonl'
    elif dataset == 'fbqa':
        inputfile = f'/xinbei_data/replug/FreebaseQA-master/{split}.jsonl'
    elif dataset == 'truqa':
        inputfile = f'/xinbei_data/replug/truthqa/truthqa{split}.jsonl'
    elif dataset == 'gaokao-geo':
        inputfile = "/xinbei_data/replug/generate/AGIEval/data/qav/gaokao-geography.jsonl"
    elif dataset == 'popqa':
        inputfile =  f'/xinbei_data/replug/generate/popqa/{split}.jsonl'
    elif dataset == 'realpopqa':
        inputfile =  f'/xinbei_data/replug/generate/popqa/realpop/{split}.jsonl'
    elif dataset == 'mmlusocial':
        inputfile = f'/xinbei_data/replug/generate/mmlu/test/socialsciences.jsonl'
    elif dataset == 'mmluhuman':
        inputfile = f'/xinbei_data/replug/generate/mmlu/test/humanities.jsonl'
    elif dataset == 'mmluother':
        inputfile = f'/xinbei_data/replug/generate/mmlu/test/other.jsonl'
    elif dataset == 'mmlustem':
        inputfile = f'/xinbei_data/replug/generate/mmlu/test/stem.jsonl'
    return inputfile

def readfiles(infile):

    if infile.endswith('json'): 
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'): 
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l, strict=False) for l in lines]
    else:
        raise NotImplementedError
    if len(lines) == 0:
        return []
    if len(lines[0]) == 1 and lines[0].get('prompt'): 
        lines = lines[1:] ## skip prompt line
    if 'answer' in lines[0].keys() and type(lines[0]['answer']) == str:
        for l in lines:
            l['answer'] = [l['answer']]
    return lines

def bing_bl(args, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/bingbl-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/bingbl(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.dataset}-{args.split}-p{args.pid}-{args.post}.jsonl'
    # search results
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.post}.jsonl'
    begin = 0
    if os.path.exists(sre): 
        print(f"Loading existing search results {sre}")
        inlines_w_sre = readfiles(sre)
        begin = len(inlines_w_sre)
        inlines[:len(inlines_w_sre)] = inlines_w_sre
    if args.nums:
        inlines = inlines[:args.nums]
    # search on bing
    fsre = open(sre, 'a')
    nowiki = 0
    if args.search == 'wiki':
        env = wikienv.WikiEnv()
    elif args.search == 'bing':
        env = BingEnv()
    for il in tqdm(range(begin, len(inlines))):
        query = 'search[' + inlines[il]['question'] +']'
        if args.search=='bing':
            # obs, reward, done, info = env.step(query, args.use_en, func='bm25', gold=inlines[il]['answer'])
            obs, reward, done, info = env.step(query, args.use_en, func=args.retrieve, gold=inlines[il]['answer'], topn=args.topn, max_words_perdoc=args.max_words_perdoc)
        else: 
            obs, reward, done, info = env.step(query)
        # obs, reward, done, info = env.step(query)
        inlines[il]['output'] = obs
        if args.sele == True and (obs.startswith("Similar:") or obs.strip == ''):
            inlines[il]['output'] = ''
            nowiki += 1
        fsre.write(json.dumps(inlines[il])+'\n')
    #ifhit
    ifhit = 0
    for line in inlines:
        sear = line['output'] if type(line['output'])==str else line['output'][0]
        ifhit += 1 if hits(line['answer'],sear) else 0
        # if hits(line['answer'],sear):
        #     print(line['answer'],sear)
    print(ifhit, ifhit /len(inlines))
    # prompt gpt
    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval 
    evalfile = f'{outputfolder}/{args.search}-{args.dataset}-metrics-{args.post}.jsonl'
    with open(evalfile, 'a') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            'nowiki': nowiki,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}: No wiki: {nowiki}')
        evalout.write(json.dumps(outmetrics) + '\n')


def step1(args, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/onlyq-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/onlyq(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.dataset}-{args.split}-p{args.pid}-{args.post}.jsonl'
    
    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval 
    evalfile = f'{outputfolder}/onlyq-{args.dataset}-metrics-{args.post}.jsonl'
    with open(evalfile, 'a', encoding='utf8') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}')
        evalout.write(json.dumps(outmetrics) + '\n')

def wiki(args, datatype, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/searchQ-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/searchQwiki(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-p{args.pid}-{args.post}.jsonl'
    # search in wikienv 
    nowiki = 0
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.post}.jsonl'
    begin = 0
    if os.path.exists(sre): 
        print(f"Loading existing search results {sre}")
        inlines_w_sre = readfiles(sre)
        begin = len(inlines_w_sre)
        inlines[:len(inlines_w_sre)] = inlines_w_sre
        if args.nums:
            inlines = inlines[:args.nums]
    
    fsre = open(sre, 'a')
    if args.search == 'wiki':
        env = wikienv.WikiEnv()
    elif args.search == 'bing':
        env = BingEnv()
    for il in tqdm(range(begin, len(inlines))):
        query = 'search[' + inlines[il]['question'] +']'
        if args.search=='bing':
            obs, reward, done, info = env.step(query, args.use_en)
        else: 
            obs, reward, done, info = env.step(query)
        # obs, reward, done, info = env.step(query)
        inlines[il]['output'] = obs
        if args.sele == True and (obs.startswith("Similar:") or obs.strip == ''):
            inlines[il]['output'] = ''
            nowiki += 1
        fsre.write(json.dumps(inlines[il])+'\n')
        # with open(sre, 'w') as f:
        #     for line in inlines:
        #         f.write(json.dumps(line)+'\n')      

    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval 
    evalfile = f'{outputfolder}/{args.search}-{args.dataset}-metrics-{args.post}.jsonl'
    with open(evalfile, 'a', encoding='utf8') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            'nowiki': nowiki,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}: No wiki: {nowiki}')
        evalout.write(json.dumps(outmetrics,ensure_ascii=False) + '\n')

def searchrewrite(args, datatype, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/searchre-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/searchre(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-p{args.pid}-{args.post}.jsonl'
    # search in wikienv 
    nowiki = 0
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.post}.jsonl'
    if os.path.exists(sre): 
        print(f"Loading existing search results {sre}")
        inlines = readfiles(sre)
        if args.nums:
            inlines = inlines[:args.nums]
    else:
        if args.search == 'wiki':
            env = wikienv.WikiEnv()
        elif args.search == 'bing':
            env = BingEnv()
        for il in tqdm(range(len(inlines))):
            query = 'search[' + inlines[il]['question'] +']'
            if args.search=='bing':
                obs, reward, done, info = env.step(query, args.use_en)
            else: 
                obs, reward, done, info = env.step(query)
            # obs, reward, done, info = env.step(query)
            inlines[il]['output'] = obs
            if args.sele == True and (obs.startswith("Similar:") or obs.strip == ''):
                inlines[il]['output'] = ''
                nowiki += 1
        with open(sre, 'w') as f:
            for line in inlines:
                f.write(json.dumps(line)+'\n') 
    # rewrite the search results
    outsr = f'{outputfolder}/{args.search}-{args.dataset}-searchrewrite-{args.post}.jsonl'
    if not os.path.exists(outsr):
        prompt_rew = "Summarize the following passage for this question, end with '**'. \n\n Question: {query} Passage: {output} \n\n Summary:"
        run_searchre(inlines, outsr, args.engine, prompt_rew, max_tokens, args.n, args.temp, args.endwith)
    rewrited = readfiles(outsr)
    for i in range(len(inlines)):
        inlines[i]['output'] = rewrited[i]['output']
    # prompt_rew = "Summarize the following passage for this question, end with '**'. \n\n Question: {query} Passage: {output} \n\n Summary:"
    # rewrited = run_searchre(inlines, outsr, args.engine, prompt_rew, max_tokens, args.n, args.temp, args.endwith)
    # for i in range(len(inlines)):
    #     inlines[i]['output'] = rewrited[i]
    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval 
    evalfile = f'{outputfolder}/{args.search}-{args.dataset}-metrics-{args.post}.jsonl'
    with open(evalfile, 'a') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            'nowiki': nowiki,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}: No wiki: {nowiki}')
        evalout.write(json.dumps(outmetrics) + '\n')

def rewrite(args, datatype, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/rewrite-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/rewrite(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-rew-p{args.pid}-{args.post}.jsonl'
    # inlines = inlines[:100]
    if os.path.exists(outputfile):
        print(f"Loading existing rewritten {outputfile}")
        re_inlines = readfiles(outputfile)
        print('begin from ', len(re_inlines))    
    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    inlines = readfiles(outputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    # search
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.pid}-{args.post}.jsonl' if not args.max_obs else f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.max_obs}-{args.pid}-{args.post}.jsonl'
    if os.path.exists(sre):
        print(f"Loading existing search results {sre}")
        srelines = readfiles(sre)
        if len(srelines) < len(inlines):
            print('continue from ', len(srelines))
            inlines = inlines[len(srelines):]
        else:
            #ifhits
            ifhit = 0
            for line in srelines:
                sear = line['output'] if type(line['output'])==str else line['output'][0]
                ifhit += 1 if hits(line['answer'],sear) else 0
                # if hits(line['answer'],sear):
                #     print(line['answer'],sear)
            print(ifhit, ifhit /len(srelines))
            return
    if args.search == 'wiki':
        env = wikienv.WikiEnv()
    elif args.search == 'bing':
        env = BingEnv()
    env.appendsimilar = True
    nowiki = 0
    sref = open(sre, 'a')
    for il in tqdm(range(len(inlines))):
        if args.think:
            # print(inlines[il])
            if "Query:" not in inlines[il]['output'][0]:
                qs= ['']
                inlines[il]['thought'] = inlines[il]['output'][0]
            else:
                qs = inlines[il]['output'][0].split("Query:")[1].split(";")
                inlines[il]['thought'] = inlines[il]['output'][0].split("Query:")[0]
        else:
            qs = inlines[il]['output'][0].split(";")
        inlines[il]['output'] = []
        for q in qs:
            print(q)
            query = 'search[' + q +']'
            if args.search=='bing':
                obs, reward, done, info = env.step(query, args.use_en, func=args.retrieve, gold=inlines[il]['answer'], topn=args.topn, max_words_perdoc=args.max_words_perdoc)
            else: 
                obs, reward, done, info = env.step(query)
            if args.sele == True and obs.startswith("Similar:"):
                inlines[il]['output'].append('')
                nowiki += 1
            else:
                inlines[il]['output'].append(obs)
                # print(obs)
        if args.max_obs:
            for o in range(len(inlines[il]['output'])):
                # word
                # print(inlines[il]['output'][o])
                trunc = " ".join(inlines[il]['output'][o].split(" ")[:args.max_obs])
                inlines[il]['output'][o] = trunc
        inlines[il]['output'] = [" ".join(inlines[il]['output'])]
        sref.write(json.dumps(inlines[il])+'\n')
        # os._exit()
    # with open(sre, 'a') as f:
    #     for line in inlines:
    #         f.write(json.dumps(line)+'\n')
    #ifhits
    ifhit = 0
    for line in inlines:
        sear = line['output'] if type(line['output'])==str else line['output'][0]
        ifhit += 1 if hits(line['answer'],sear) else 0
    print(ifhit, ifhit /len(inlines))

def rewrite2(args, datatype, max_tokens, prompt):
    searchre = False
    # for hotpot
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/rewrite-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/rewrite(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.repid}-{args.post}.jsonl' if not args.max_obs else f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.max_obs}-{args.repid}-{args.post}.jsonl'
    inlines = readfiles(sre)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/rewrite2con-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/rewrite2con(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-rew2con-p{args.pid}-{args.post}-{args.repid}.jsonl' if not args.max_obs else f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-rew2con-p{args.pid}-{args.post}-{args.max_obs}-{args.repid}.jsonl'
    
    if searchre == True:
        outsr = f'{outputfolder}/{args.search}-{args.dataset}-searchrewrite-{args.post}.jsonl'
        if not os.path.exists(outsr):
            prompt_rew = "Summarize the following passage for this question, end with '**'. \n\n Question: {query} Passage: {output} \n\n Summary:"
            run_searchre(inlines, outsr, args.engine, prompt_rew, max_tokens, args.n, args.temp, args.endwith)
        rewrited = readfiles(outsr)
        for i in range(len(inlines)):
            inlines[i]['output'] = rewrited[i]['output']
    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval 
    evalfile = f'{outputfolder}/{args.search}-rewrite2con-{args.dataset}-metrics-{args.repid}-{args.post}.jsonl' if not args.max_obs else f'{outputfolder}/{args.search}-rewrite2con-{args.dataset}-{args.post}-metrics-{args.max_obs}-{args.repid}.jsonl'
    with open(evalfile, 'a') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}')
        evalout.write(json.dumps(outmetrics) + '\n')

def goldsearch(args):
    # need question, searchresult, ans-> a folder
    allfiles = os.listdir(args.goldtest)
    answer_file = ''
    for ifil in allfiles:
        print(ifil)
        if 'searchresult' in str(ifil):
            if str(args.pid) in str(ifil) or args.post in str(ifil):
                search_res_file = ifil
                s = readfiles(os.path.join(args.goldtest,search_res_file))
        if args.split in str(ifil) and str(args.pid) in str(ifil):
            answer_file = ifil
            a = readfiles(os.path.join(args.goldtest, answer_file))
    print('answer: ', answer_file, search_res_file)
    if 'rew' in answer_file or not len(answer_file):
        dir_ = args.goldtest.replace("rewrite-",'rewrite2con-')
        allfiles = os.listdir(dir_)
        for ifil in allfiles:
            if args.split in str(ifil) and str(args.pid) in str(ifil) and str(args.repid) in str(ifil) and args.post in str(ifil):
                answer_file = ifil
                a = readfiles(os.path.join(dir_, answer_file))
    search_res_file = 'bing-realpopqa-searchresult-12-pt-testset.jsonl'
    answer_file = 'bing-realpopqa-test-rew2con-p2-pt-12.jsonl'
    dir_ = args.goldtest.replace("rewrite-",'rewrite2con-')
    print('answer: ', answer_file, search_res_file)
    s = readfiles(os.path.join(args.goldtest,search_res_file))
    a = readfiles(os.path.join(dir_, answer_file))
    print(len(s), len(a))
    # re = re[:len(a)]
    s = s[:len(a)]
    all = []
    f1s = []
    hit = []
    searchlen = []
    for i, (sr, ans) in tqdm(enumerate(zip(s, a)), total=len(s)):
        # f1_ = f1(ans['output'][0],ans['answer'])
        # f1s.append(f1_)
        search_res_ = sr['output'] if len(sr['output'])!= 1 else sr['output'][0]
        hit_ = hits(ans['answer'], search_res_)
        hit.append(hit_)
        all_ = {
            'question': ans['question'],
            # 'rewrite': q['output'],
            'answer' : ans['answer'],
            'output' : ans['output'],
            # 'em': ems(ans['output'][0],ans['answer']),
            # 'f1': f1_,
            'hit': hit_,
            'retrieved': sr['output'],
            # 'retrieved_len' : len
            # (sr['output'].split())
        }
        searchlen.append(len(search_res_.split()))
        if hit_ > 0:
            all.append(all_)
    # print(searchlen)
    searchlenavg= sum(searchlen)/len(s)
    print(len(all))
    outputf = args.goldtest + '/hited.jsonl'
    with open(outputf, 'w') as evalout:
        for iall in all:
            evalout.write(json.dumps(iall) + '\n')
    evalfile = args.goldtest + '/hited_metrics.jsonl'
    with open(evalfile, 'a') as evalout:
        emscore, length, f1 = eval_question_answering(outputf, args.endwith)
        outmetrics = {
            'outputfile': outputf,
            # 'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            "search": args.search,
            'nums':args.nums,
            'hit': len(all_),
            'searchlen': searchlenavg
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}; hit: {len(all)}; searchlen:{searchlenavg}')
        evalout.write(json.dumps(outmetrics) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--dataset", default=None, type=str, required=True,
        help="dataset name: [nq, tqa, webq, wizard, fever, fm2]",
    )
    parser.add_argument("--task", default='step1', type=str, required=True,
        help="task name: [step1, wiki, rewrite, rewrite2], should be either 1 or 2",
    )
    parser.add_argument("--split", default='test', type=str, required=True,
        help="dataset split: [train, dev, test]",
    )
    parser.add_argument("--engine", default='chatgpt', type=str, required=False,
        help="engine/deployment id",
    )
    parser.add_argument("--n", default=1, type=int, required=False, help='--num_sequence')
    parser.add_argument("--temp", default=0, type=float, required=False, help='--temperature')
    parser.add_argument('--pid', default=1, type=int, required=True)
    parser.add_argument('--endwith', default=None, type=str)
    parser.add_argument('--sele', action='store_true')
    parser.add_argument('--promptfile', default='myprompt', type=str)
    parser.add_argument('--search', type=str, default='wiki', help='bing / wiki')
    parser.add_argument('--nums', type=int, default=None)
    parser.add_argument('--post', type=str, default='pt', help='postfix')
    parser.add_argument('--max_obs', type=int, default=None)
    parser.add_argument('--repid', type=int, default=None)
    parser.add_argument('--think', action='store_true')
    parser.add_argument('--output_dir',type=str, default='./outputs')
    parser.add_argument('--use_en', action="store_true")
    parser.add_argument('--retrieve', type=str, default='plain', help='[plain, bm25, rdoc, gold]')
    parser.add_argument('--topn', type=int, default=10)
    parser.add_argument('--max_words_perdoc', type=int, default=800)
    parser.add_argument('--goldtest', type=str)

    args = parser.parse_args()
    args.endwith = '**'
    # args.endwith = None
    datatype = 'question answering'
    max_tokens = 300 # answer max length

    promptfile = args.promptfile
    promptlines = open(f'inprompts/{promptfile}.jsonl', 'r').readlines()
    print("Using the prompt file:\n", promptfile)

    for line in promptlines:
        line = json.loads(line)

        if line['type'] == datatype and line['task'] == args.task and line['pid'] == args.pid:
            print("Using the prompt template:\n", line['prompt'])
            prompt = line['prompt']
            pid = line['pid']

            if args.task == 'step1': # only use question
                outputs = step1(args, max_tokens, prompt)
            elif args.task == 'wiki':
                outputs = bing_bl(args,  max_tokens, prompt)
            elif args.task == 'searchre':
                outputs = searchrewrite(args, datatype, max_tokens, prompt)
            elif args.task == 'rewrite':
                outputs = rewrite(args, datatype, max_tokens, prompt)
            elif args.task == 'rewrite2':
                outputs = rewrite2(args, datatype, max_tokens, prompt)
            # elif args.task == 'goldtest':
            #     outputs = rewrite2(args, max_tokens, prompt)
            else:  ## should be either 1 or 2
                raise NotImplementedError
            
            if promptfile == 'regular':
                break ## only use the first prompt 
    # no prompt
    if args.task == 'goldtest':
        outputs = goldsearch(args)