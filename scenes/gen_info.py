import json
from random import choice
import os

with open('scene_configs/value_name_new.json', 'r') as f:
    v = json.load(f)


path = './outputs/commands'
for dir in os.listdir(path):
    c_dir = os.path.join(path, dir)
    for seed in os.listdir(c_dir):
        child_dir = os.path.join(c_dir, seed)
        values_low = []
        values_high = []
        with open(child_dir+'/log.json', 'r') as f:
            command = json.load(f)
        with open(child_dir+'/info.json', 'r') as f:
            info = json.load(f)
        target = info['target']
        for item in command:
            if item['$type'] == 'add_object':
                name = item['name']
                if name in v.keys() and item['id'] not in target:
                    value = v[name]
                    if value:
                        values_high.append(item['id'])
                    else :
                        values_low.append(item['id'])
        ret = target
        if len(values_low)>0:
            ret.append(choice(values_low))
        if len(values_high)>0:
            ret.append(choice(values_high))
        if len(ret) > 4:
            info['target'] = info['target'][:4]
        elif len(ret) < 4:
            print(ret)
        with open(child_dir+'/info.json', 'w') as f:
            json.dump(info, f)