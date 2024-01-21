import os
import json
import pdb
from argparse import ArgumentParser
from tdw.controller import Controller

parser = ArgumentParser()
parser.add_argument('-i', "--input", type=str, default="./outputs/commands")
parser.add_argument('-o', "--output", type=str, default="../data/room_setup_fire/test_set_new_object")
args = parser.parse_args()

data_dir = args.input
newdata_dir = args.output

flood_json = {"source": [[1.647413730621338, 1.5, 1.5]], "direction": [[45, 0, 0]], "speed": [5], "flood_source_from": "x_max"}

def work(setup, seed):
    D = os.path.join(data_dir, setup, seed)
    dirs = os.listdir(D)
    info_f = "info.json"
    commands_f = dirs[0] if dirs[0] != info_f else dirs[1]

    info_f = open(os.path.join(D, info_f), "r")
    commands_f = open(os.path.join(D, commands_f), "r")
    info = json.load(info_f)
    commands = json.load(commands_f)
    info_f.close()
    commands_f.close()

    # print info type, commands type
    filtered_commands = []
    targets = info["target"]
    targets_category = []
    for command in commands:
        tp = command["$type"]
        if tp == "terminate":
            break
        if tp[:4] == "send" or tp.find("avatar") != -1 or "avatar_id" in command:
            continue
        if "url" in command:
            if tp == "add_object":
                try:
                    new_url = \
                    Controller.get_add_object(model_name=command["name"], object_id=1, library="models_full.json")[
                        "url"]
                except:
                    new_url = \
                    Controller.get_add_object(model_name=command["name"], object_id=1, library="models_special.json")[
                        "url"]
                if command['id'] in targets and command['category'] not in targets_category:
                    targets_category.append(command['category'])
            elif tp == "add_scene":
                new_url = Controller.get_add_scene(scene_name=command["name"])["url"]
            # print(command["url"], new_url)
            command["url"] = new_url
        filtered_commands.append(command)
    print(info)
    newinfo = {}
    newinfo["task"] = "fire"
    newinfo["containers"] = []
    newinfo["agent"] = info["agent"]
    newinfo["targets"] = targets_category
    newinfo["other"] = dict(
        fire=info["fire"],
    )
    newdir = os.path.join(newdata_dir, f"{setup}-{seed}")
    os.makedirs(newdir, exist_ok=True)
    with open(os.path.join(newdir, "info.json"), "w") as f:
        json.dump(newinfo, f, indent=4)
    with open(os.path.join(newdir, "log.txt"), "w") as f:
        json.dump(filtered_commands, f)
    with open(os.path.join(newdir, "flood.json"), "w") as f:
        json.dump(flood_json, f)


# work("mm_craftroom_2a", "47")
setups = os.listdir(data_dir)
for setup in setups:
    seeds = os.listdir(os.path.join(data_dir, setup))
    for seed in seeds:
        work(setup, seed)
