import sys
import os
PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until the folder name is HAZARD
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)
sys.path.append(PATH)

from envs.wind.wind_gym import WindEnv
from envs.wind.agent import *
from tdw.output_data import Raycast
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.replicant.ik_plans.ik_plan_type import IkPlanType
from tdw.output_data import OutputData
from src.HAZARD.policy.astar import get_astar_path

class PathMarker(AddOn):
    def __init__(self):
        super().__init__()
        self.last_marked_len = 0

    def get_initialization_commands(self) -> List[dict]:
        return []

    def on_send(self, resp: List[bytes]) -> None:
        return None

    def mark_path(self, path:Optional[List[List[int]]] = None):
        for i in range(self.last_marked_len):
            self.commands.append({"$type": "remove_position_markers", "id": 1919810 + i})
        if path is None:
            return
        for (i, point) in enumerate(path):
            self.commands.append({"$type": "add_position_marker", "position": TDWUtils.array_to_vector3(point), "id": 1919810 + i, "scale": 0.1})
        self.last_marked_len = len(path)

class AgentCameraTracker(AddOn):
    def __init__(self, replicant: Replicant, camera: ThirdPersonCamera, relative_position: np.ndarray = [6, 6, 0]):
        super().__init__()
        self.replicant = replicant
        self.camera = camera
        self.relative_position = relative_position
        self.initialized = True
    
    def get_initialization_commands(self) -> List[dict]:
        return super().get_initialization_commands()
    
    def on_send(self, resp: List[bytes]) -> None:
        position = self.replicant.dynamic.transform.position + self.relative_position
        self.camera.teleport(position=TDWUtils.array_to_vector3(position))
        self.camera.look_at(target=TDWUtils.array_to_vector3(self.replicant.dynamic.transform.position))
        return None

class BoxCastOccupancyMap(AddOn):
    def __init__(self):
        super().__init__()
        self.grid: Optional[np.ndarray] = None
        self.origin: Optional[np.ndarray] = None
        self.grid_size: Optional[np.ndarray] = None
        self.num_grid: Optional[List[int]] = None
        self.initialized = True
        self.floor_height: Optional[float] = None
        self.remembered_commands = []
        self.frequency = "never"
    
    def get_initialization_commands(self) -> List[dict]:
        return []
    
    def on_send(self, resp: List[bytes]) -> None:
        self.grid = np.zeros(self.num_grid, dtype=int)
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
                            self.grid[idx // self.num_grid[1], idx % self.num_grid[1]] = 1
                    else:
                        self.grid[idx // self.num_grid[1], idx % self.num_grid[1]] = 100
        if self.frequency == "always":
            self.commands = self.remembered_commands.copy()
    
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

    def generate(self, grid_size: float = 0.25, boundX = [-10, 10], boundZ = [-10, 10], floor_height = 0.0,
                 frequency:str = "once", ignore:Optional[List[int]] = None) -> None:
        self.grid_size = grid_size
        self.num_grid = [int((boundX[1] - boundX[0]) / grid_size) + 5, int((boundZ[1] - boundZ[0]) / grid_size) + 5]
        self.origin = [int(-boundX[0] / grid_size) + 2, int(-boundZ[0] / grid_size) + 2]
        self.floor_height = floor_height
        self.frequency = frequency

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
        if self.frequency == "always":
            self.remembered_commands = self.commands.copy()

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
    
    def nav_path(self, origin, destination):
        origin = self.real_to_grid(origin)
        destination = self.real_to_grid(destination)
        if origin[0] < 0 or origin[0] >= self.num_grid[0] or origin[1] < 0 or origin[1] >= self.num_grid[1]:
            return None
        if destination[0] < 0 or destination[0] >= self.num_grid[0] or destination[1] < 0 or destination[1] >= self.num_grid[1]:
            return None
        mask = np.ones(self.num_grid)
        mask[max(origin[0]-1, 0):min(origin[0]+2, self.num_grid[0]), max(origin[1]-1, 0):min(origin[1]+2, self.num_grid[1])] = 1
        weight = np.ones(self.num_grid) + self.grid * 10000 * mask
        for i in range(self.num_grid[0]):
            for j in range(self.num_grid[1]):
                s = (self.grid * mask)[max(i-1, 0):min(i+2, self.num_grid[0]), max(j-1, 0):min(j+2, self.num_grid[1])].sum()
                weight[i, j] += s * 5
        weight = weight.astype(np.float32)
        grid_path = get_astar_path(weight, origin, destination)
        # grid_path = path_cleanup(grid_path)
        real_path = []
        for p in grid_path:
            real_path.append(self.grid_to_real(p))
        return real_path

occ = BoxCastOccupancyMap()
marker = PathMarker()

def find_path(env: WindEnv, target: int, reset_arms: bool = False):
    agent_pos = env.controller.agents[0].dynamic.transform.position
    occ.generate(boundX=[agent_pos[0] - 10, agent_pos[0] + 10], boundZ=[agent_pos[2] - 10, agent_pos[2] + 10])
    env.controller.communicate([])
    start = env.controller.agents[0].dynamic.transform.position
    end = env.controller.manager.objects[target].position
    path = occ.nav_path(start, end)
    
    if path is None:
        return "Bad!"
    # mark points in path
    marker.mark_path(path)
    for point in path:
        p = TDWUtils.array_to_vector3(point)
        p["y"] = 0
        env.controller.do_action(agent_idx=0, action="move_to", params={"target": p,
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.25})
        env.controller.next_key_frame()
    env.controller.communicate([])


def run(scene_name, image_path=""):
    env = WindEnv(launch_build=True, screen_size=512, port=12138)
    # env = WindEnv(launch_build=False, screen_size=512, port=1071, use_local_resources=True)
    data_dir = os.path.join(PATH, "data", "room_setup_wind", scene_name)
    env.reset(data_dir=data_dir)

    # delete this folder (non-empty)
    if os.path.exists(image_path):
        os.system(f"rm -rf {image_path}")
    os.makedirs(image_path, exist_ok=True)
    camera = ThirdPersonCamera(avatar_id=f"{scene_name}", position={"x": 0, "y": 8, "z": 8}, look_at={"x": 0, "y": 0, "z": 0})
    ic = ImageCapture(path=image_path, avatar_ids=[f"{scene_name}", str(env.controller.agents[0].replicant_id)], pass_masks=["_img", "_depth", "_id"])
    env.controller.add_ons.extend([camera, ic, occ, marker])
    env.controller.add_ons.append(AgentCameraTracker(env.controller.agents[0], camera))
    occ.generate()
    env.controller.communicate([])

    def navigate():
        nearest = env.controller.find_nearest_object()
        if nearest == None:
            return "done"
        if env.controller.frame_count > 2000:
            return "max_steps_reached"
        print("target object: ", nearest)
        
        if find_path(env, nearest) == "Bad!":
            return "failed"
        # env.controller.do_action(agent_idx=0, action="move_to", params={"target": nearest,
        #                                                                 "reset_arms": True,
        #                                                                 "arrived_at": 0.5})
        # print(env.controller.next_key_frame(), env.controller.agents[0]._previous_action)

        env.controller.do_action(agent_idx=0, action="reach_for", params={"target": nearest, 
                                                                        "arm": [Arm.left, Arm.right]})
        print(env.controller.next_key_frame(), env.controller.agents[0]._previous_action)
        
        # env.controller.communicate({"$type": "teleport_object", "id": nearest, "position": {"x": -3, "y": 0, "z": -3}})

        env.controller.do_action(agent_idx=0, action="grasp", params={"target": nearest, 
                                                                        "arm": Arm.left,
                                                                        "angle": None,
                                                                        "axis": None,})
        print(env.controller.next_key_frame(), env.controller.agents[0]._previous_action)
        if env.controller.agents[0].action.status != ActionStatus.success:
            return "failed"

        env.controller.do_action(agent_idx=0, action="reset_arm", params={"arm": [Arm.left, Arm.right]})
        print(env.controller.next_key_frame(), env.controller.agents[0]._previous_action)
        

        target = env.controller.find_nearest_container()
        print("target container: ", target)
        if find_path(env, target) == "Bad!":
            env.controller.do_action(agent_idx=0, action="drop", params={"arm": Arm.left})
            print(env.controller.next_key_frame(), env.controller.agents[0]._previous_action)
            env.controller.manager.settled.remove(nearest)
            return "failed"
        
        env.controller.do_action(agent_idx=0, action="turn_to", params={"target": target})
        print(env.controller.next_key_frame(), env.controller.agents[0]._previous_action)
        
        obj = env.controller.manager.objects[target]
        top = obj.top() + [0, 0.1, 0]
        env.controller.do_action(agent_idx=0, action="reach_for", params={"target": TDWUtils.array_to_vector3(top),
                                                                          "absolute": True,
                                                                          "offhand_follows": False,
                                                                          "arm": Arm.left,
                                                                          "from_held": True,
                                                                          "held_point": "top",
                                                                          "max_distance": 1.5,
                                                                          "plan": IkPlanType.vertical_horizontal})
        print(env.controller.next_key_frame(), env.controller.agents[0]._previous_action)

        env.controller.manager.settled.add(nearest)
        
        env.controller.do_action(agent_idx=0, action="drop", params={"arm": Arm.left})
        print(env.controller.next_key_frame(), env.controller.agents[0]._previous_action)
        
        env.controller.manager.settled.remove(nearest)

        env.controller.do_action(agent_idx=0, action="reset_arm", params={"arm": [Arm.left, Arm.right]})
        print(env.controller.next_key_frame(), env.controller.agents[0]._previous_action)
        
        return "success"

    num_pickable = len(env.setup.resistence)
    num_picked = -len(env.controller.manager.settled)
    failed_trials = 0
    while True:
        resp = navigate()
        print("settled:", env.controller.manager.settled, "resp:", resp)
        if resp == "success":
            continue
        elif resp == "failed":
            failed_trials += 1
            if failed_trials >= 30:
                break
            continue
        else:
            break
    num_picked += len(env.controller.manager.settled)
    fout = open(f"{PATH}/logs/result.txt", "a")
    import time
    # fout.write(f"{scene_name}: {num_picked}/{num_pickable}\n")
    fout.write(f"{scene_name}: {num_picked}/{num_pickable} {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}\n")
    fout.close()
    for i in range(200):
        env.controller.communicate([])

    env.controller.communicate({"$type": "terminate"})
    env.controller.socket.close()

    os.system(f"ffmpeg -i {image_path}/{scene_name}/img_%04d.jpg -vcodec libx264 -pix_fmt yuv420p {PATH}/logs/{scene_name}.mp4 < {PATH}/policy/yes 2>/dev/null")
