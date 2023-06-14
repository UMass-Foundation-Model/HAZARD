import os
import subprocess
from time import sleep
from random import randint
scenes = ["mm_craftroom_2a", "mm_craftroom_3a", "mm_kitchen_2a", "mm_kitchen_3a"]
scenes = scenes[1:2]


for s in scenes:
    seed = randint(0, 100)
    seed = 76
    cnt = 0
    for d in range(seed, seed+8):
        p = subprocess.Popen('DISPLAY=:4.0 nice -n 19 /data/private/zqh/embodied/tdw/TDW/TDW.x86_64', shell=True)
        sleep(1）
        q = subprocess.run('nice python ./procgen_object_placement.py -d '+str(d)+' -s '+s, shell=True)
        p.wait()
        print(s+f' {cnt} finished')
        cnt+=1
