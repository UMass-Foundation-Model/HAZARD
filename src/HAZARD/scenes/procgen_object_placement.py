import pdb

from tdw.controller import Controller
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.floorplan import Floorplan
from tdw.tdw_utils import TDWUtils
import numpy as np
from tdw.librarian import SceneLibrarian, ModelLibrarian
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.scene_data.scene_bounds import SceneBounds
from utils import *
import json

from argparse import ArgumentParser
from tdw.tdw_utils import TDWUtils

# import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

parser = ArgumentParser()
parser.add_argument('-d', "--seed", type=int, default=1)
parser.add_argument('-s', "--scene", type=str, default='mm_kitchen_3a')
parser.add_argument('-n', "--with_new_objects", action='store_true', default=False)
args = parser.parse_args()

scenes = args.scene
seed = args.seed
np.random.seed(seed)

# TDWUtils.set_default_libraries(scene_library="../../local_asset_linux/scenes.json",
#                                model_library="../../local_asset_linux/models.json")

# c = Controller(launch_build=False)
c = Controller()

occ = OccupancyMap()
occ.generate(cell_size=0.25, once=False)
om = ObjectManager(transforms=True, rigidbodies=True, bounds=True)
c.add_ons.extend([occ, om])

print(scenes)
procgenkitchen = ProcGenKitchen()
procgenkitchen.create(scene=scenes)
# Choose from: 
# ['mm_craftroom_2a', 'mm_craftroom_2b', 'mm_craftroom_3a', 'mm_craftroom_3b', 
#  'mm_kitchen_2a
commands_init_scene = [{"$type": "set_screen_size", "width": 1280, "height": 720}] # Set screen size
commands_init_scene.extend(procgenkitchen.commands)


response = c.communicate(commands_init_scene)
print("ProcGenKitchen init completed. ")


d = {}
with open('./scene_configs/list_new.json' if args.with_new_objects else './scene_configs/list.json', 'r') as f:
    d = json.loads(f.read())

l = []
commands = []
stuff = ['table', 'chair']
types = ['positive', 'negative']
candidate_targets = []
candidate_name = []
cnt = 1
for object_id in om.objects_static:
    if om.objects_static[object_id].category in stuff: 
        obj = c.get_unique_id()
        name = np.random.choice(d[types[cnt]])
        candidate_targets.append(obj)
        candidate_name.append(name)
        p = shift(om.bounds[object_id])
        l.append(p)
        cnt = (cnt+1)%2
        commands.append(c.get_add_object(name,
                                    object_id=obj,
                                    position=TDWUtils.array_to_vector3(p),
                                    ))
    
for object_id in om.objects_static:
    p = om.bounds[object_id].bottom
    ret = dis(p, l)
    if ret>0:
        del l[ret]
        del commands[ret]
        del candidate_targets[ret]
        del candidate_name[ret]
    
print("Object placement on surfaces completed. ")

# Place different objects on the floor of different rooms. (For ProcGenKitchen, there is only one room)


# Read object list
random_object_list_on_floor = [[] for i in range(1)] # The number of rooms in ProcGenKitchen is 1. 
with open("./scene_configs/random_object_list_on_floor_new.txt" if args.with_new_objects else
          "./scene_configs/random_object_list_on_floor.txt") as f:
    for obj_name in f.readlines():
        for lst in random_object_list_on_floor:
            lst.append(obj_name[:-1].replace("\n", ""))

# Try to add custom objects on the ground
object_probability = 0.05

for x_index in range(occ.occupancy_map.shape[0]):
    for z_index in range(occ.occupancy_map.shape[1]):
        if occ.occupancy_map[x_index][z_index]==0: # Unoccupied
            if np.random.random()<object_probability:
                x = float(occ.positions[x_index][z_index][0])
                z = float(occ.positions[x_index][z_index][1])
                room_id = 0 # Only one room in ProcGenKitchen. 
                obj_type = np.random.choice(random_object_list_on_floor[room_id])
                obj = c.get_unique_id()
                candidate_name.append(obj_type)
                candidate_targets.append(obj)
                print(obj_type, obj)
                commands.append(c.get_add_object(obj_type,
                                                 object_id=obj,
                                                 position={"x": x, 
                                                           "y": 0, 
                                                           "z": z},
                                                 rotation={"x": 0, 
                                                           "y": np.random.uniform(0, 360), 
                                                           "z": 0}, 
                                                 ))

commands.append({"$type": "step_physics", "frames": 100})
print("Object placement on floor completed. ")

# %%
# Camera settings and screenshots

camera = ThirdPersonCamera(position={"x": -1.5, "y": 3, "z": -1.5},
                           look_at={"x": 2.5, "y": 0, "z": 2.5},
                           avatar_id="a")



screenshot_path = "./outputs/screenshots/" + scenes + '/' + str(seed) + '/'
print(f"Images will be saved to: {screenshot_path}")
capture = ImageCapture(avatar_ids=["a"], path=screenshot_path, pass_masks=["_img"])
commands.extend(camera.get_initialization_commands()) # Init camera
c.add_ons.extend([capture]) # Capture images
print("Camera setup completed. ")

# Save commands to json

import datetime
import json
import os
PATH = os.path.dirname(os.path.abspath(__file__))
dirname = os.path.join(PATH, "outputs/commands", scenes ,f'{seed}')
os.makedirs(dirname, exist_ok=True)
# command_save_path = os.path.join(dirname, "ProcGenKitchen %s.json"%(datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")))
command_save_path = os.path.join(dirname, "log.json")
with open(command_save_path, "w") as f:
    f.write(json.dumps(commands_init_scene+commands, indent=4))
# c.communicate({"$type": "terminate"})
print("Scene setup commands saved to %s. "%command_save_path)

# Send all commands to the controller


om.initialized = False
response = c.communicate(commands)
print("TDW scene setup completed. ")

grid, origin, grid_size = generate_grid(occ)

# select the agent, fire, target
free_positions = [(x, y) for x in range(0, len(grid)) for y in range(0, len(grid[0])) if grid[x][y] == 2]
free_positions = np.array(free_positions)
target = np.random.choice(candidate_targets)
target_position = real_to_grid(om.bounds[target].bottom, origin, grid_size)
# nearest free
_ = np.linalg.norm(free_positions - target_position, axis=1)
target_position = free_positions[np.argmin(_)]
# agent position is > 0.5 max distance from target
dist_target = BFS(grid, target_position)
max_dist = 0
for i in range(dist_target.shape[0]):
    for j in range(dist_target.shape[1]):
        if dist_target[i, j] > max_dist and dist_target[i, j] < 1e8:
            max_dist = dist_target[i, j]
agent_position = []
for i in range(dist_target.shape[0]):
    for j in range(dist_target.shape[1]):
        if dist_target[i, j] > 0.5 * max_dist and dist_target[i, j] < 1e8:
            agent_position.append([i, j])
agent_position = agent_position[np.random.choice(len(agent_position))]
dist_agent = BFS(grid, agent_position)
# fire position is those dist_target + dist_agent < real distance + 1, meaning it will be near the path and both are reachable
fire_position = np.argwhere(dist_target + dist_agent < dist_target[agent_position[0], agent_position[1]] + 1)
num_fire = np.random.randint(1, min(4, len(fire_position)))
fire_position = fire_position[np.random.choice(len(fire_position), num_fire, replace=False), :].tolist()
# encode agent, fire, target into a json file
agent_position = grid_to_real(agent_position, origin, grid_size)
target_position = grid_to_real(target_position, origin, grid_size)
fire_position = [grid_to_real(fire, origin, grid_size) for fire in fire_position]


import json
values = []
t_val = 0
with open('./scene_configs/value_name_new.json' if args.with_new_objects else
          './scene_configs/value_name.json', 'r') as f:
    d = json.load(f)
i = 0
for index, obj in enumerate(candidate_targets):
    if obj == target:
        i = index
        break
t_name = candidate_name[i]
if d[t_name] == 1:
    t_val = 1

for index, obj in enumerate(candidate_targets):
    name = candidate_name[index]
    if d[name]!=t_val:
        values.append(index)

if len(values) == 0:
    values = np.random.choice(candidate_targets)
else:
    values = candidate_targets[np.random.choice(values)]


with open(os.path.join(dirname, "info.json"), "w") as f:
    data = {"agent": agent_position, "target": [int(target), values], "fire": fire_position}
    print(data)
    json.dump(data, f)

c.communicate({"$type": "terminate"})


