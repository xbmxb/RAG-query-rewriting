import json, os, random
def split(path):
    data = []
    lines = open(path, 'r', encoding='utf8').readlines()
    data = [json.loads(l, strict=False) for l in lines]
    print("total: ", len(data))
    trainp = path.split(".")[0]+'-train'+'.jsonl'
    testp = path.split(".")[0]+'-test'+'.jsonl'
    train = []
    test = []
    for d in data:
        r = random.uniform(0,1)
        if r <= 0.2:
            test.append(d)
        else:
            train.append(d)
    with open(trainp,'w') as f:
        for d in train:
            json.dump(d, f)
            f.write('\n')
    with open(testp,'w') as f:
        for d in test:
            json.dump(d, f)
            f.write('\n')


def addans(path):
    lines = open(path, 'r', encoding='utf8').readlines()
    data = [json.loads(l, strict=False) for l in lines]
    data_ = []
    for d in data:
        opts = []
        opts.append(d['question'].split('D.')[-1].strip())
        opts.append(d['question'].split('D.')[0].split("C.")[-1].strip())
        opts.append(d['question'].split('D.')[0].split("C.")[0].split("B.")[-1].strip())
        opts.append(d['question'].split('D.')[0].split("C.")[0].split("B.")[0].split("A.")[-1].strip())
        # print(opts)
        # assert len(opts) == 5
        if d['answer'] == 'A':
            d['answer'] = ['A', 'A. '+opts[3]]
        elif d['answer'] == 'B':
            d['answer'] = ['B', 'B. '+opts[2]]
        elif d['answer'] == 'C':
            d['answer'] = ['C', 'C. '+opts[1]]
        elif d['answer'] == 'D':
            d['answer'] = ['D', 'D. '+opts[0]]
        else:
            print('error: ', d['answer'])
        data_.append(d)
    path_ = path.split(".")[0]+'-add.jsonl'
    with open(path_,'w') as f:
        for d in data_:
            json.dump(d, f)
            f.write('\n')

# split('socialsciences.jsonl')
# split('other.jsonl')
# split('humanities.jsonl')
# split('stem.jsonl')
# addans('mysplit/socialsciences-test.jsonl')
addans('mysplit/socialsciences-train.jsonl')
addans('mysplit/humanities-test.jsonl')
addans('mysplit/humanities-train.jsonl')
addans('mysplit/other-test.jsonl')
addans('mysplit/other-train.jsonl')
addans('mysplit/stem-test.jsonl')
addans('mysplit/stem-train.jsonl')
