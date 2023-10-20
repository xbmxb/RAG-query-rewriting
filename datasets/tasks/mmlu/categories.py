import json
import pandas as pd
import random
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

datas = {
    "STEM": [],
    "humanities": [],
    "social sciences": [],
    "other (business, health, misc.)": [],
}
choices = ['A','B','C','D']
for dataf in subcategories.keys():
    # print(dataf)
    fail = 0
    # f=open('../data/test/'+dataf+'_test.csv', 'r')
    test_df = pd.read_csv('../data/test/'+dataf+'_test.csv', header=None)
    
    data = []
    linec = 0
    for idx in range(test_df.shape[0]):
        question = test_df.iloc[idx, 0]
        k = test_df.shape[1] - 2
        choice = ''
        for j in range(k):
            choice += "{}. {}  ".format(choices[j], test_df.iloc[idx, j+1])
        ans = test_df.iloc[idx, k + 1]
        # ans = line.split(',')[-1].strip()
        # if ans not in ['a','b','c','d','A','B','C','D']:
        #     fail += 1
        # question = ','.join(line.split(',')[:-1]).strip()
        # question = line.split('\"')[1]
        # options = line.split('\"')[-1].split(',')
        # try:
        #     assert len(options) == 4
        # except:
        #     print(len(options))
        #     print(line)
        # question = question + 'Options: A.' + options[0] + ' B.'+options[1] + ' C.'+options[2] + ' D.' +options[3]
        d = {
            'question': question + 'Options: ' + choice,
            'answer': ans
        }
        data.append(d)
        linec += 1
    # print(linec)
    for cat in categories.keys():
        if subcategories[dataf][0] in categories[cat]:
            print(dataf,'-->',cat)
            datas[cat].extend(data)

for j in datas.keys():
    print(j, len(datas[j]))
    with open(j+'.jsonl','w') as f:
        for d in datas[j]:
            json.dump(d, f)
            f.write('\n')

def split(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.load(line))
    print("total: ", len(data))
    trainp = path.split(".")[0]+'train'+'.jsonl'
    testp = path.split(".")[0]+'test'+'.jsonl'
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

split('socialsciences.jsonl')