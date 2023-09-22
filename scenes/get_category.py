import json
from random import choice
import os

with open('scene_configs/value_name.json', 'r') as f:
    v = json.load(f)

with open('../../local_asset_linux/models.json', 'r') as f:
    d = json.load(f)
d = d['records']

category = {}
cnt=0
for key, value in v.items():
    cate = d[key]['wcategory']
    if cate not in category.keys():
        category[cate] = value
    # category[cate].append(value)
print(category)
with open('scene_configs/value.json', 'w') as f:
    json.dump(category, f, indent=4)