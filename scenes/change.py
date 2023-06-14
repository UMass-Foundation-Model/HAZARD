import json
from random import choice
import os


path = './outputs/commands'
for dir in os.listdir(path):
    c_dir = os.path.join(path, dir)
    for seed in os.listdir(c_dir):
        child_dir = os.path.join(c_dir, seed)
        with open(child_dir+'/info_old.json', 'r') as f:
            info = json.load(f)
        with open(child_dir+'/log.json', 'r') as f:
            command = json.load(f)
        obj = {}
        for item in command:
            if item['$type'] == 'add_object':
                obj[item['id']] = item['name']
        info['other'] = {'fire': info['fire']}
        info.pop('fire')
        t_name = set()
        for i in info['target']:
            t_name.add(obj[i])
        info['targets'] = list(t_name)
        info.pop('target')
        with open(child_dir+'/info.json', 'w') as f:
            json.dump(info, f)