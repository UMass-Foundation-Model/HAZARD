import os
import subprocess
from time import sleep
from random import randint
from tqdm import tqdm
scenes = ["mm_craftroom_2a", "mm_craftroom_3a", "mm_kitchen_2a", "mm_kitchen_3a"]
# scenes = scenes[1:2]
# scenes = scenes[2:]


for s in scenes:
    seed = randint(0, 100)
    seed = 114
    cnt = 0
    for d in tqdm(range(seed, seed+100)):
        dirname = os.path.join("./outputs/commands", s, f'{d}')
        if os.path.exists(dirname):
            print(f"{dirname} already exists. Skipping. ")
            continue
        q = subprocess.run('python ./procgen_object_placement.py -d '+str(d)+' -s '+s, shell=True)
        print(s+f' {d} finished, {cnt} of total {100}')
        cnt+=1
