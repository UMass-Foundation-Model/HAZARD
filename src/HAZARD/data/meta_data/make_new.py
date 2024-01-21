import json
import os
old = open("categories.txt")
old_cate = eval(old.read())
BASE_PATH = "../room_setup_fire/test_set_new_object"
dirs = os.listdir(BASE_PATH)
for scene in dirs:
    info = json.load(open(os.path.join(BASE_PATH, scene, "info.json")))
    target = info['targets']
    for t in target:
        if t not in old_cate:
            old_cate.append(t)
new = open("categories_new.txt", "w")
cate_str = ", ".join([f"'{cate}'" for cate in old_cate])
new.write(f"[{cate_str}]")
