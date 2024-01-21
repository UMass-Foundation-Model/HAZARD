from tdw.controller import Controller
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.floorplan import Floorplan
from tdw.tdw_utils import TDWUtils
from tdw.librarian import SceneLibrarian, ModelLibrarian
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.scene_data.scene_bounds import SceneBounds
from utils import *
from random import choice
import json

# TDWUtils.download_asset_bundles(path="./local_asset_bundles",
#                                 models={"models_core.json": ["iron_box"]},
#                                 scenes={"scenes.json": ["tdw_room", "mm_craftroom_3b"]})
# local_scene_lib = SceneLibrarian(library="./local_asset_bundles/scenes.json")
# local_model_lib = ModelLibrarian(library="./local_asset_bundles/models.json")

import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

c = Controller()

occ = OccupancyMap()
occ.generate(cell_size=0.25, once=False)
om = ObjectManager(transforms=True, rigidbodies=True, bounds=True)
c.add_ons.extend([occ, om])

floorplan = Floorplan()
FLOORPLAN_SCENE_NAME = "2b"
FLOORPLAN_LAYOUT = 1
floorplan.init_scene(scene=FLOORPLAN_SCENE_NAME, layout=FLOORPLAN_LAYOUT)

commands_init_scene = [{"$type": "set_screen_size", "width": 1280, "height": 720}] # Set screen size
commands_init_scene.extend(floorplan.commands)
commands_init_scene.append({"$type": "set_floorplan_roof", "show": False}) # Hide roof

response = c.communicate(commands_init_scene)


resp = c.communicate([{"$type": "send_scene_regions"}])
scene_bounds = SceneBounds(resp=resp)



d = {}
with open('scene_configs/list.json', 'r') as f:
    d = json.loads(f.read())

commands = []
for object_id in om.objects_static:
    # print(object_id, om.objects_static[object_id].name, om.objects_static[object_id].category)
    id = belongs_to_which_room(om.bounds[object_id].top[0], om.bounds[object_id].top[2], scene_bounds)
    func_name = get_room_functional_by_id(FLOORPLAN_SCENE_NAME, FLOORPLAN_LAYOUT, id)
    if om.objects_static[object_id].category == 'table': 
        obj = c.get_unique_id()
        p = shift(om.bounds[object_id])
        commands.append(c.get_add_object(choice(d[om.objects_static[object_id].category][func_name]),
                                    object_id=obj,
                                    position=TDWUtils.array_to_vector3(p),
                                    ))
    if om.objects_static[object_id].category == 'sofa': 
        obj = c.get_unique_id()
        p = shift(om.bounds[object_id])
        commands.append(c.get_add_object(choice(d[om.objects_static[object_id].category][func_name]),
                                    object_id=obj,
                                    position=TDWUtils.array_to_vector3(p),
                                    ))
    if om.objects_static[object_id].category == 'chair': 
        obj = c.get_unique_id()
        p = shift(om.bounds[object_id])
        commands.append(c.get_add_object(choice(d[om.objects_static[object_id].category][func_name]),
                                    object_id=obj,
                                    position=TDWUtils.array_to_vector3(p),
                                    ))
    if om.objects_static[object_id].category == 'bed': 
        obj = c.get_unique_id()
        p = shift(om.bounds[object_id])
        commands.append(c.get_add_object(choice(d[om.objects_static[object_id].category][func_name]),
                                    object_id=obj,
                                    position=TDWUtils.array_to_vector3(p),
                                    ))


import numpy as np

# Read object list
random_object_list_on_floor = [[] for i in range(get_total_rooms(FLOORPLAN_SCENE_NAME))]
with open("scene_configs/random_object_list_on_floor.txt") as f:
    for obj_name in f.readlines():
        for lst in random_object_list_on_floor:
            lst.append(obj_name[:-1].replace("\n", ""))
# print(random_object_list_on_floor)

# Try to add custom objects on the ground
object_probability = 0.01

for x_index in range(occ.occupancy_map.shape[0]):
    for z_index in range(occ.occupancy_map.shape[1]):
        if occ.occupancy_map[x_index][z_index]==0: # Unoccupied
            if np.random.random()<object_probability:
                x = float(occ.positions[x_index][z_index][0])
                z = float(occ.positions[x_index][z_index][1])
                room_id = belongs_to_which_room(x, z, scene_bounds)
                if room_id<0: 
                    continue
                obj_type = np.random.choice(random_object_list_on_floor[room_id])
                commands.append(c.get_add_object(obj_type,
                                                 object_id=c.get_unique_id(),
                                                 position={"x": x, 
                                                           "y": 0, 
                                                           "z": z},
                                                 rotation={"x": 0, 
                                                           "y": np.random.uniform(0, 360), 
                                                           "z": 0}, 
                                                 ))
commands.append({"$type": "step_physics", "frames": 100})

# Camera settings and screenshots

camera = ThirdPersonCamera(position={"x": 0, "y": 30, "z": 0},
                           look_at={"x": 0, "y": 0, "z": 0},
                           avatar_id="a")
screenshot_path = "./outputs/screenshots/"
print(f"Images will be saved to: {screenshot_path}")
capture = ImageCapture(avatar_ids=["a"], path=screenshot_path, pass_masks=["_img"])
commands.extend(camera.get_initialization_commands()) # Init camera
c.add_ons.extend([capture]) # Capture images

# Save commands to json

import datetime
import json
import os
os.makedirs("./outputs/commands/", exist_ok=True)
with open("./outputs/commands/%s.json"%(datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")), "w") as f:
    f.write(json.dumps(commands_init_scene+commands, indent=4))
# c.communicate({"$type": "terminate"})

# Send all commands to the controller

response = c.communicate(commands)

c.communicate({"$type": "terminate"})