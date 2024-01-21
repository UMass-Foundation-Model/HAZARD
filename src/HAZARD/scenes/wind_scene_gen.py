from tdw.add_ons.logger import Logger
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.add_on import AddOn
from tdw.output_data import OutputData, Raycast
from tdw.add_ons.log_playback import LogPlayback
import random
import numpy as np
import os
from typing import List, Optional

import json

"""
0: unoccupied
1: occupied
100: no floor
"""

class BoxCastOccupancyMap(AddOn):
    def __init__(self):
        super().__init__()
        self.grid: Optional[np.ndarray] = None
        self.origin: Optional[np.ndarray] = None
        self.grid_size: Optional[np.ndarray] = None
        self.num_grid: Optional[List[int]] = None
        self.initialized = True
        self.floor_height: Optional[float] = None
    
    def get_initialization_commands(self) -> List[dict]:
        return []
    
    def on_send(self, resp: List[bytes]) -> None:
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "rayc":
                rayc = Raycast(resp[i])
                idx = rayc.get_raycast_id()
                if idx >= 114514 and idx < 114514 + self.num_grid[0] * self.num_grid[1]:
                    idx -= 114514
                    if rayc.get_hit():
                        hit_y = rayc.get_point()[1]
                        # print("hit point=", rayc.get_point(), "i, j=", idx // self.num_grid[1], idx % self.num_grid[1])
                        if hit_y > self.floor_height + 0.01:
                            self.grid[idx // self.num_grid[1]][idx % self.num_grid[1]] = 1
                    else:
                        self.grid[idx // self.num_grid[1]][idx % self.num_grid[1]] = 100
    
    def grid_to_real(self, position):
        if not isinstance(position, list):
            position = position.tolist()
        return [position[0] * self.grid_size - self.origin[0] * self.grid_size, self.floor_height, position[1] * self.grid_size - self.origin[1] * self.grid_size]

    def real_to_grid(self, position):
        if not isinstance(position, list):
            position = position.tolist()
        if len(position) > 2:
            position = [position[0], position[2]]
        return [int((position[0] + self.origin[0] * self.grid_size + 0.01) / self.grid_size), int((position[1] + self.origin[1] * self.grid_size + 0.01) / self.grid_size)]

    def generate(self, grid_size: float = 0.25, boundX = [-8, 8], boundZ = [-8, 8], floor_height = 0.0) -> None:
        self.grid_size = grid_size
        self.num_grid = [int((boundX[1] - boundX[0]) / grid_size) + 5, int((boundZ[1] - boundZ[0]) / grid_size) + 5]
        self.origin = [int(-boundX[0] / grid_size) + 2, int(-boundZ[0] / grid_size) + 2]
        self.floor_height = floor_height

        self.grid = np.zeros(self.num_grid, dtype=int)
        for i in range(self.num_grid[0]):
            for j in range(self.num_grid[1]):
        # for i in range(22, 23):
        #     for j in range(22, 23):
                start = np.array(self.grid_to_real([i, j])) - [0, 20, 0]
                end = start + [0, 40, 0]
                # print(start, end, i, j)
                self.commands.append({"$type": "send_boxcast",
                                      "half_extents": {"x": grid_size / 2, "y": 0, "z": grid_size / 2},
                                      "origin": TDWUtils.array_to_vector3(end),
                                      "destination": TDWUtils.array_to_vector3(start),
                                      "id": i * self.num_grid[1] + j + 114514})
    def find_free(self, r):
        candidates = []
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                s = self.grid[i-r:i+r+1, j-r:j+r+1].sum()
                if s == 0:
                    candidates.append([i, j])
        if len(candidates) == 0:
            return None
        pos = random.choice(candidates)
        return self.grid_to_real(pos)

global container_list, object_list
def load_config():
    global container_list, object_list
    with open("scene_configs/container_list.json", "r") as f:
        container_list = json.load(f)
    with open("scene_configs/object_list.json", "r") as f:
        object_list = json.load(f)

PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until the folder name is HAZARD
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)

available_scenes = ["suburb_scene_2023"]

floor_heights = {
    "abandoned_factory": 0.0,
    "ruin": 0.0,
    "downtown_alleys": 0.0,
    "suburb_scene_2023": 0.02,
    "iceland_beach": 0.0,
    "lava_field": 5
}
object_scales = {
    "bag": 1.0,
    "backpack": 0.7,
    "purse": 2.0,
    "basket": 1.0,
    "money": 2.0,
    "bottle": 3.0,
    "box": 0.5,
    "electronic": 2.0,
}

num_objects = 15
num_containers = 3

"""
55 0
-58 -9
58 -51
-61 55"""
position_config = [
    [0, 0],
    [55, 0],
    [-58, -9],
    [58, -51],
    [-61, 55],
]

def generate(scene_name, p, seed):
    np.random.seed(seed)
    random.seed(seed)
    floor_height = floor_heights[scene_name]
    origin = position_config[p]

    c = Controller(launch_build=True, port=12138)
    logger = Logger(path=os.path.join(PATH, "data", "room_setup_wind", f"{scene_name}-{p}-{seed}", "log.txt"))
    occ = BoxCastOccupancyMap()
    occ.generate(floor_height=floor_height, boundX=[-8 + origin[0], 8 + origin[0]], boundZ=[-8 + origin[1], 8 + origin[1]])
    c.add_ons.append(logger)
    c.add_ons.append(occ)

    commands = [c.get_add_scene(scene_name=scene_name), {"$type": "set_screen_size", "width": 512, "height": 512}]

    camera = ThirdPersonCamera(avatar_id="abra", position={"x": 5, "y": 8, "z": 5}, look_at={"x": 0, "y": 5, "z": 0})
    c.add_ons.append(camera)
    c.communicate(commands)
    # occ.generate(floor_height=floor_height)
    # c.communicate([c.get_add_object(model_name="b04_shoppping_cart", position=TDWUtils.array_to_vector3([0, 3, 0]), object_id=1, library="models_full.json")])

    # fout = open("grid.txt", "w")
    # for i in range(occ.grid.shape[0]):
    #     for j in range(occ.grid.shape[1]):
    #         if occ.grid[i][j] == 0:
    #             fout.write(".")
    #         elif occ.grid[i][j] == 1:
    #             fout.write("X")
    #         elif occ.grid[i][j] == 100:
    #             fout.write(" ")
    #     fout.write("\n")
    # fout.close()
    
    # input()
    # c.communicate({"$type": "terminate"})
    # c.socket.close()
    # return

    # put containers
    config = dict()
    containers = []
    for _ in range(num_containers):
        container_name = random.choice(container_list["cart"])
        pos = occ.find_free(5)
        if pos is None:
            continue
        idx = c.get_unique_id()
        commands = [c.get_add_object(model_name=container_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
        commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
        # commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([1.5, 1, 1.5])})
        
        containers.append(idx)

        occ.generate(floor_height=floor_height, boundX=[-8 + origin[0], 8 + origin[0]], boundZ=[-8 + origin[1], 8 + origin[1]])
        c.communicate(commands)
    
    wind_resistence = dict()
    for _ in range(num_objects):
        object_type = random.choice(list(object_list.keys()))
        object_description = random.choice(object_list[object_type])
        object_name = object_description["name"]
        object_wind_resistence = object_description["wind_resistence"]

        pos = occ.find_free(4)
        if pos is None:
            continue
        idx = c.get_unique_id()
        wind_resistence[idx] = object_wind_resistence
        pos[1] += 0.3
        commands = [c.get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
        scale = object_scales[object_type]
        commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([scale, scale, scale])})
        
        occ.generate(floor_height=floor_height, boundX=[-8 + origin[0], 8 + origin[0]], boundZ=[-8 + origin[1], 8 + origin[1]])
        c.communicate(commands)
    
    # agent
    agent_pos = occ.find_free(1)
    if agent_pos is None:
        print("Bad!")
        c.communicate({"$type": "terminate"})
        c.socket.close()
    
    # config["agent"] = agent_pos
    # config["containers"] = containers
    # rad = random.random() * 2 * np.pi
    # magnitude = random.random() * 3 + 3
    # config["wind"] = [np.cos(rad) * magnitude, 0, np.sin(rad) * magnitude]
    # config["wind_resistence"] = wind_resistence
    config["task"] = "wind"
    config["containers"] = containers
    config["agent"] = agent_pos
    config["targets"] = list(wind_resistence.keys())
    rad = random.random() * 2 * np.pi
    magnitude = random.random() * 5 + 5
    config["other"] = dict(
        wind_resistence=wind_resistence,
        wind=[np.cos(rad) * magnitude, 0, np.sin(rad) * magnitude]
    )

    info_path = os.path.join(PATH, "data", "room_setup_wind", f"{scene_name}-{p}-{seed}")
    os.makedirs(info_path, exist_ok=True)
    info = open(os.path.join(PATH, "data", "room_setup_wind", f"{scene_name}-{p}-{seed}", "info.json"), "w")
    json.dump(config, info)
    info.close()
    c.communicate({"$type": "step_physics", "frames": 50})
    c.communicate({"$type": "terminate"})
    c.socket.close()
    
    # clean up the log file
    playback = LogPlayback()
    playback.load(os.path.join(PATH, "data", "room_setup_wind", f"{scene_name}-{p}-{seed}", "log.txt"))
    
    logger.reset(path=os.path.join(PATH, "data", "room_setup_wind", f"{scene_name}-{p}-{seed}", "log.txt"))
    c = Controller(launch_build=True, port=12138)
    c.add_ons.append(logger)
    for a in playback.playback:
        commands = []
        for cc in a:
            tp = cc["$type"]
            if tp.find("send") != -1 or tp.find("avatar") != -1 or "avatar_id" in cc:
                continue
            commands.append(cc)
        c.communicate(commands)
    c.socket.close()

if __name__ == "__main__":
    load_config()
    # print(container_list, object_list)
    configs = []
    for scene_name in available_scenes:
        for p in range(len(position_config)):
            for seed in range(20):
                configs.append((scene_name, p, seed))

    import tqdm
    for scene_name, p, seed in tqdm.tqdm(configs):
        generate(scene_name, p, seed)