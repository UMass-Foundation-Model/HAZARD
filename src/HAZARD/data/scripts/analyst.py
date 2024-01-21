import json
import os

num = []
category = set()

def get_num(t, l):
    num=0
    for i in l:
        if i['$type']=='add_object' and i['category']==t:
            num+=1
    return num

path = '../room_setup_wind'
for dir in os.listdir(path):
    if dir[0]!='s':
        continue
    child_dir = os.path.join(path, dir)
    # d = {}
    # with open(child_dir + '/log.txt', 'r') as f:
    #     for l in f:
    #         log = json.loads(l)
    #         log = log[0]
    #         if log['$type'] == 'add_object':
    #             d[log['id']] = log['category']
    with open(child_dir + '/info.json', 'r') as f:
        info = json.load(f)
    target = info['targets']
    num.append(len(target))

print(num)
print(max(num), min(num), sum(num)/len(num))