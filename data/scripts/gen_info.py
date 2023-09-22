import json
from random import choice
import os

with open('../meta_data/value.json', 'r') as f:
    v = json.load(f)


path = '../room_setup'
for dir in os.listdir(path):
    child_dir = os.path.join(path, dir)
    if os.path.isfile(child_dir):
        continue
    values_low = []
    values_high = []
    with open(child_dir+'/log.txt', 'r') as f:
        for i in range(2):
            _ = f.readline()
        line = f.readline()
        text = json.loads(line)
        l = []
        for item in text:
            if item['$type'] == 'add_object':
                name = item['name']
                if name in v.keys():
                    value = v[name]
                    if value:
                        values_high.append(item['id'])
                    else :
                        values_low.append(item['id'])
                else:
                    print(name)
    ret = []
    if len(values_low)>0:
        ret.append(choice(values_low))
    if len(values_high)>0:
        ret.append(choice(values_high))
    if len(ret)!=2:
        print(ret)
    ret = {'target':ret}
    with open(child_dir+'/info.json', 'r') as f:
        info = json.load(f)
    info.update(ret)
    with open(child_dir+'/info.json', 'w') as f:
        json.dump(info, f)

