import json
sp = 'test.source'
tp = 'test.target'

sf = open(sp,'r')
tf = open(tp, 'r')
s = []
t = []
for line in sf:
    s.append(line)
for line in tf:
    t.append(line)
data = []
for si, ti in zip(s,t):
    data.append({'target': ti, 'source': si})
with open('test.json','w') as f:
    json.dump(data,f, indent=4)
