import os
import json
import pdb

from tdw.controller import Controller

data_dir = "scenes/outputs/commands"
newdata_dir = "data/room_setup_fire"


def work(setup, seed, new_order, test=False):
    D = os.path.join(data_dir, setup, seed)
    dirs = os.listdir(D)
    info_f = "info.json"
    commands_f = "log.json"

    info_f = open(os.path.join(D, info_f), "r")
    commands_f = open(os.path.join(D, commands_f), "r")
    info = json.load(info_f)
    commands = json.load(commands_f)
    info_f.close()
    commands_f.close()

    # print info type, commands type
    filtered_commands = []
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
    newinfo["targets"] = info["targets"]
    newinfo["other"] = dict(
        fire=info["other"]["fire"],
    )
    if test:
        newdir = os.path.join(newdata_dir, f"test_set/{setup}-{str(new_order+1)}")
    else:
        newdir = os.path.join(newdata_dir, f"{setup}-{str(new_order+1)}")
    os.makedirs(newdir, exist_ok=True)
    with open(os.path.join(newdir, "info.json"), "w") as f:
        json.dump(newinfo, f, indent=4)
    with open(os.path.join(newdir, "log.txt"), "w") as f:
        json.dump(filtered_commands, f)


# work("mm_craftroom_2a", "47")
setups = os.listdir(data_dir)
for setup in setups:
    seeds = os.listdir(os.path.join(data_dir, setup))
    if setup == "mm_kitchen_3a":
        for new_order, seed in enumerate(seeds):
            work(setup, seed, new_order, test=True)
    else:
        for new_order, seed in enumerate(seeds):
            work(setup, seed, new_order)
