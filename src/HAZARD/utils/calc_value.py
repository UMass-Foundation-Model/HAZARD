import sys
import os
from envs.flood.utils import ObjectState as FloodObjectState
from envs.fire.fire_utils import ObjectState as FireObjectState
from src.HAZARD.utils.scene_setup import SceneSetup
import tqdm

if len(sys.argv) > 2:
    dirname = sys.argv[1]
    taskname = sys.argv[2]
print("dirname: ", dirname)

tasks = ["flood", "fire", "wind"]

scenes = os.listdir(dirname)
scenes = [scene for scene in scenes if os.path.isdir(os.path.join(dirname, scene))]

data_dirs = {
    "fire": "room_setup_fire",
    "wind": "room_setup_wind",
    "flood": "room_setup_fire"
}

def get_values(taskname, scene, object_id, states):
    waterproof_dict = json.load(open("src/HAZARD/scenes/scene_configs/fluid.json"))
    value_dict = json.load(open("src/HAZARD/data/meta_data/value.json"))
    data_dir = data_dirs[taskname]
    if os.path.exists(os.path.join("data", data_dir, "test_set", scene)):
        scene_setup = SceneSetup(os.path.join("data", data_dir, "test_set", scene))
    else:
        scene_setup = SceneSetup(os.path.join("data", data_dir, scene))
    # print(scene_setup.target_id2name, scene)
    name = scene_setup.target_id2name[object_id]
    if name in value_dict:
        if value_dict[name] == 1:
            value = 5
        else:
            value = 1
    else:
        value = 1
    discount = 1
    try:
        if taskname == "fire":
            state = FireObjectState(states[str(object_id)])
            if state == FireObjectState.NORMAL:
                discount = 1
            else:
                discount = 0.5
        elif taskname == "flood":
            state = FloodObjectState(states[str(object_id)])
            waterproof = 0
            if name in waterproof_dict:
                waterproof = waterproof_dict[name]
            if state == FloodObjectState.NORMAL or waterproof == 1:
                discount = 1
            else:
                discount = 0.5
        elif taskname == "wind":
            discount = 1
    except:
        pass
    return value * discount, value, discount

sum_rate = 0
sum_value_rate = 0
sum_step = 0
sum_picked = 0
sum_damaged = 0
valid_scenes = []

for scene in tqdm.tqdm(scenes):
    if scene[-3:] == "log":
        continue
    fname = os.path.join(dirname, scene, "eval_result.json")
    if not os.path.exists(fname):
        continue
    valid_scenes.append(scene)
    import json
    with open(fname, "r") as f:
        data = json.load(f)
    
    target_status = data["target_status"]
    final_states = dict()
    if "final_states" in data:
        final_states = data["final_states"]

    tot_targets = 0
    max_step = 0
    tot_value = 0
    tot_picked = 0
    tot_max_value = 0
    tot_dmg = 0

    for idx, status in target_status.items():
        idx = int(idx)
        tot_targets += 1
        value, max_value, dmg = get_values(taskname, scene, idx, final_states)
        tot_max_value += max_value
        if status == False:
            continue
        if dmg < 0.6:
            tot_dmg += 1
        tot_picked += 1
        max_step = max(max_step, int(status))
        tot_value += value
    
    rate = tot_picked * 1.0 / tot_targets
    value_rate = tot_value * 1.0 / tot_max_value
    sum_picked += tot_picked
    sum_step += max_step
    sum_rate += rate
    sum_value_rate += value_rate
    sum_damaged += tot_dmg

print("task name", dirname)
print(len(valid_scenes))
print("Average rate", sum_rate / len(valid_scenes))
print("Average value rate", sum_value_rate / len(valid_scenes))
print("Average step", sum_step / sum_picked)
print("Damage rate", sum_damaged / sum_picked)

# print("Average step: ", sum_step / len(scenes))
# print("Average picked: ", sum_picked / len(scenes))
# print("Average value: ", sum_value / len(scenes))
# print("tot picked", sum_picked)
# print("Average rate", sum_rate / len(scenes))