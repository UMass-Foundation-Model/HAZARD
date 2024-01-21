
from tdw.controller import Controller
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.floorplan import Floorplan
from tdw.tdw_utils import TDWUtils
from tdw.librarian import SceneLibrarian, ModelLibrarian
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.logger import Logger

import os
import random
import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until the folder name is HAZARD
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)

"""
| `scene` | `layout` |
| --- | --- |
| 1a, 1b, or 1c | 0, 1, or 2 |
| 2a, 2b, or 2c | 0, 1, or 2 |
| 4a, 4b, or 4c | 0, 1, or 2 |
| 5a, 5b, or 5c | 0, 1, or 2 |
"""

available_scenes = ["1a", "1b", "1c", "2a", "2b", "2c", "4a", "4b", "4c", "5a", "5b", "5c"]
available_layouts = [0, 1, 2]

def shift(bounds):
    eps = 1e-3
    y = bounds.top[1]
    z1 = bounds.back[-1]+eps
    z2 = bounds.front[-1]-eps
    x1 = bounds.left[0]+eps
    x2 = bounds.right[0]-eps
    z = z1 + random.random()*(z2 - z1)
    x = x1 + random.random()*(x2 - x1)
    return x, y, z

def BFS(grid: np.ndarray, start):
    dist = np.ones(grid.shape) * 1e9
    dist[start[0]][start[1]] = 0
    q = [start]
    while len(q) > 0:
        u = q.pop(0)
        for v in [(u[0] + 1, u[1]), (u[0] - 1, u[1]), (u[0], u[1] + 1), (u[0], u[1] - 1)]:
            if v[0] >= 0 and v[0] < len(grid) and v[1] >= 0 and v[1] < len(grid[0]):
                if dist[v[0]][v[1]] > 1e8 and grid[v[0]][v[1]] == 2:
                    dist[v[0]][v[1]] = dist[u[0]][u[1]] + 1
                    q.append(v)
    return dist

def generate_grid(occ):
    """
    grid = 0: not in the room
    grid = 1: occupied
    grid = 2: free
    """
    boundX = [np.min(occ.positions[:, :, 0]), np.max(occ.positions[:, :, 0])]
    boundZ = [np.min(occ.positions[:, :, 1]), np.max(occ.positions[:, :, 1])]
    grid_size = 0.25
    num_grid = [int((boundX[1] - boundX[0]) / grid_size) + 5, int((boundZ[1] - boundZ[0]) / grid_size) + 5]
    origin = [int(-boundX[0] / grid_size) + 2, int(-boundZ[0] / grid_size) + 2]

    grid = np.zeros(num_grid, dtype=int)
    for i in range(len(occ.occupancy_map)):
        for j in range(len(occ.occupancy_map[i])):
            if occ.occupancy_map[i][j] == 1:
                p = occ.positions[i, j]
                grid[int(p[0] / grid_size) + origin[0]][int(p[1] / grid_size) + origin[1]] = 1
            else:
                p = occ.positions[i, j]
                grid[int(p[0] / grid_size) + origin[0]][int(p[1] / grid_size) + origin[1]] = 2

    # f = open("occ.txt", "w")
    # for i in range(len(grid)):
    #     for j in range(len(grid[i])):
    #         if grid[i][j] == 1:
    #             f.write("x")
    #         else:
    #             f.write(" ")
    #     f.write("\n")
    # f.close()
    return grid, origin, grid_size

def grid_to_real(position, origin, grid_size):
    if not isinstance(position, list):
        position = position.tolist()
    return [position[0] * grid_size - origin[0] * grid_size, 0.0, position[1] * grid_size - origin[1] * grid_size]

def real_to_grid(position, origin, grid_size):
    if not isinstance(position, list):
        position = position.tolist()
    if len(position) > 2:
        position = [position[0], position[2]]
    return [int((position[0] + origin[0] * grid_size) / grid_size), int((position[1] + origin[1] * grid_size) / grid_size)]

def generate(scene="1a", layout=0, seed=114514):
    dirname = os.path.join(PATH, "data", f"{scene}-{layout}-{seed}")
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    logger = Logger(path=os.path.join(dirname, "log.txt"))
    random.seed(seed)
    np.random.seed(seed)

    c = Controller()
    c.add_ons.append(logger)
    # c.communicate([{"$type": "set_screen_size", "width": 1280, "height": 720}])

    occ = OccupancyMap()
    occ.generate(cell_size=0.25, once=False)

    om = ObjectManager(transforms=True, rigidbodies=True, bounds=True)

    floorplan = Floorplan()
    floorplan.init_scene(scene=scene, layout=layout)

    c.add_ons.extend([floorplan, occ, om])
    c.communicate([])

    c.communicate([{"$type": "set_floorplan_roof", "show": False}])
    
    commands = []

    candidate_targets = []
    for object_id in om.objects_static:
        # print(object_id, om.objects_static[object_id].name, om.objects_static[object_id].category)
        if om.objects_static[object_id].category == 'table': # 在正确的物体上放置要添加的东西
            obj = c.get_unique_id()
            candidate_targets.append(obj)
            p = shift(om.bounds[object_id])
            commands.append(c.get_add_object('apple',
                                        object_id=obj,
                                        position=TDWUtils.array_to_vector3(p),
                                        ))
        if om.objects_static[object_id].category == 'sofa': 
            obj = c.get_unique_id()
            candidate_targets.append(obj)
            p = shift(om.bounds[object_id])
            commands.append(c.get_add_object('red_bag',
                                        object_id=obj,
                                        position=TDWUtils.array_to_vector3(p),
                                        ))
        if om.objects_static[object_id].category == 'chair': 
            obj = c.get_unique_id()
            candidate_targets.append(obj)
            p = shift(om.bounds[object_id])
            commands.append(c.get_add_object('backpack',
                                        object_id=obj,
                                        position=TDWUtils.array_to_vector3(p),
                                        ))
        if om.objects_static[object_id].category == 'bed': 
            obj = c.get_unique_id()
            candidate_targets.append(obj)
            p = shift(om.bounds[object_id])
            commands.append(c.get_add_object('pillow',
                                        object_id=obj,
                                        position=TDWUtils.array_to_vector3(p),
                                        ))
    
    # Read object list
    random_object_list_on_floor = []
    with open(os.path.join(PATH, "scene", "scene_configs", "random_object_list_on_floor.txt")) as f:
        for obj_name in f.readlines():
            if obj_name[-1]=="\n":
                random_object_list_on_floor.append(obj_name[:-1])
            else:
                random_object_list_on_floor.append(obj_name)
    # print(random_object_list)

    # Try to add custom objects on the ground
    object_probability = 0.01

    for x_index in range(occ.occupancy_map.shape[0]):
        for z_index in range(occ.occupancy_map.shape[1]):
            if occ.occupancy_map[x_index][z_index]==0: # Unoccupied
                if np.random.random()<object_probability:
                    obj_type = np.random.choice(random_object_list_on_floor)
                    obj = c.get_unique_id()
                    candidate_targets.append(obj)
                    commands.append(c.get_add_object(obj_type,
                                                    object_id=obj,
                                                    position={"x": float(occ.positions[x_index][z_index][0]), 
                                                            "y": 0, 
                                                            "z": float(occ.positions[x_index][z_index][1])},
                                                    rotation={"x": 0, 
                                                            "y": np.random.uniform(0, 360), 
                                                            "z": 0}, 
                                                    ))
    commands.append({"$type": "step_physics", "frames": 100})

    # camera = ThirdPersonCamera(position={"x": 0, "y": 30, "z": 0},
    #                        look_at={"x": 0, "y": 0, "z": 0},
    #                        avatar_id="a")
    # path = os.path.join(dirname, "images")
    # print(f"Images will be saved to: {path}")
    # capture = ImageCapture(avatar_ids=["a"], path=path, pass_masks=["_img"])
    # c.add_ons.extend([camera, capture]) # Capture images

    om.initialized = False
    c.communicate(commands)

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
    with open(os.path.join(dirname, "info.json"), "w") as f:
        data = {"agent": agent_position, "target": int(target), "fire": fire_position}
        print(data)
        json.dump(data, f)
    
    # # add agent
    # from tdw.add_ons.replicant import Replicant
    # commands = []
    # agent = Replicant(replicant_id=c.get_unique_id(), position=TDWUtils.array_to_vector3(agent_position), name="fireman")
    # c.add_ons.append(agent)
    # c.add_agent(agent.replicant_id, np.array(agent_position))
    # for fire in fire_position:
    #     commands.append(c.get_add_visual_effect("fire", effect_id=c.get_unique_id(), position=TDWUtils.array_to_vector3(fire)))
    
    # c.communicate(commands)
    # for i in range(100):
    #     c.communicate([])
    c.communicate({"$type": "terminate"})

if __name__ == "__main__":
    generate(scene="1a", layout=0, seed=114514)
