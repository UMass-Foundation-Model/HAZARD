import gym
import gym.spaces
from envs.wind.windagent_controller import *
from envs.wind.agent import *
import numpy as np
from policy.env_actions import visualize_obs, agent_drop, agent_pickup, agent_walk_to, agent_explore, agent_walk_to_single_step

from enum import IntEnum

import os
PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until it contains envs folder
while not os.path.exists(os.path.join(PATH, "envs")):
    PATH = os.path.dirname(PATH)

class ActionSpace(IntEnum):
    WALK_TO_NEAREST_TARGET = 0
    WALK_TO_NEAREST_CONTAINER = 1
    PICK_UP_NEAREST = 2
    DROP = 3
    EXPLORE = 4
    WALK_TO_RANDOM_OBJECT_IN_SIGHT = 5

class WindEnv(gym.Env):
    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = False, seed = 0,
                 screen_size = 512, use_local_resources = False, map_size_h=256, map_size_v=256, grid_size=0.25,
                 image_capture_path = None, **kwargs):
        self.controller_args = dict(launch_build=launch_build, port=port, check_version=check_version,
                                    screen_size=screen_size, use_local_resources=use_local_resources,
                                    map_size_h=map_size_h, map_size_v=map_size_v, grid_size=grid_size,
                                    image_capture_path=image_capture_path)
        self.controller = None
        self.RNG = np.random.RandomState(0)

        # rgb_space = gym.spaces.Box(0, 256, (3, screen_size, screen_size), dtype=np.float)
        # seg_space = gym.spaces.Box(0, 256, (1, screen_size, screen_size), dtype=np.float)
        # depth_space = gym.spaces.Box(0, 256, (1, screen_size, screen_size), dtype=np.float)
        # object_space = gym.spaces.Dict({
        #     "object_id": gym.spaces.Discrete(30),
        #     "type": gym.spaces.Discrete(4),
        #     "seg_color": gym.spaces.Box(0, 256, (3,), dtype=np.int32),
        # })
        # temperature_space = gym.spaces.Box(0, 10000, (1, screen_size, screen_size), dtype=np.float64)
        self.done = False

        self.observation_space = gym.spaces.Box(0, 20, (4, map_size_h, map_size_v), dtype=np.float64)
        # self.observation_space = gym.spaces.Dict({
        #     "rgb": rgb_space,
        #     "seg_mask": seg_space,
        #     "depth": depth_space,
        #     "temperature": temperature_space,
        #     "agent": gym.spaces.Box(-30, 30, (6, ), dtype=np.float32),
        #     "status": gym.spaces.Discrete(3),
        #     'camera_matrix': gym.spaces.Box(-30, 30, (4, 4), dtype=np.float32)
        # })

        self.action_space = gym.spaces.Discrete(6)
        self.max_step = 1000
    
    def reset(self, data_dir = None):
        if data_dir == None:
            data_dirs = os.listdir(os.path.join(PATH, "data", "room_setup_wind"))
            data_dirs = [d for d in data_dirs if "test" not in d]
            data_dir = os.path.join(PATH, "data", "room_setup_wind", data_dirs[self.RNG.randint(len(data_dirs))])
        self.setup = SceneSetup(data_dir=data_dir)
        if self.controller is not None:
            self.controller.communicate({"$type": "terminate"})
            self.controller.socket.close()
        self.controller = WindAgentController(**self.controller_args)
        self.controller.seed(self.RNG.randint(1000000))
        print("Controller connected")
        self.controller.init_scene(self.setup)
        self.controller.do_action(0, "turn_by", {"angle": 0})
        self.controller.next_key_frame()

        self.num_step = 0
        self.last_action = None
        self.last_target = None
        return self.controller._obs()["RL"]
    
    # def action2command(self, action, agent_idx: int = 0):
    #     command = None
    #     if action == ActionSpace.MOVE_FORWARD:
    #         command = ["move_by", {"distance": 0.5}]
    #     elif action == ActionSpace.TURN_LEFT:
    #         command = ["turn_by", {"angle": -15.0}]
    #     elif action == ActionSpace.TURN_RIGHT:
    #         command = ["turn_by", {"angle": 15.0}]
    #     elif action == ActionSpace.PICK_UP:
    #         target = self.controller.find_nearest_object(agent_idx)
    #         command = ["pick_up", {"target": target, "arm": [Arm.left, Arm.right],
    #                                "angle": None, "axis": None}]
    #     elif action == ActionSpace.DROP:
    #         command = ["drop", {"arm": Arm.left}]
    #     elif action == ActionSpace.STOP:
    #         command = ["stop", {}]
    #     return command
    
    def step(self, action):
        # if self.done:
        #     self.done = False
        #     self.reset()
        """
        for each type, if action_target is 0, ignore the action target
        """
        if not isinstance(action, int):
            action = action.item()
        # action_type, action_target = action // 50, action % 50
        # action_type = int(action_type)
        # # action_target = int(action_target)
        # # action_target = self.controller.manager.get_real_id(action_target)
        
        reward = 1
        # if action_type == ActionSpace.WALK_TO:
        #     if action_target is None:
        #         result, msg = False, "invalid target"
        #         reward -= 100
        #     else:
        #         result, msg = agent_walk_to_single_step(self, target=action_target)
        # elif action_type == ActionSpace.PICK_UP:
        #     if action_target is None or not action_target in self.controller.targets or action_target:
        #         result, msg = False, "invalid target"
        #         reward -= 100
        #     else:
        #         result, msg = agent_pickup(self, target=action_target)
        # elif action_type == ActionSpace.DROP:
        #     if action_target is None or not action_target in self.controller.containers:
        #         reward -= 100
        #         result, msg = False, "invalid target"
        #     else:
        #         result, msg = agent_drop(self, container=action_target)
        # elif action_type == ActionSpace.EXPLORE:
        #     if action_target is not None:
        #         result, msg = False, "explore should not have target"
        #         reward -= 100
        #     else:
        #         result, msg = agent_explore(self)
        # else:
        #     result, msg = False, "what?" # this should not happen at all
        result, msg = None, None
        target = None
        if action == ActionSpace.WALK_TO_NEAREST_TARGET:
            targets = [idx for idx in self.controller.targets if idx not in self.controller.finished]
            target = self.controller.find_nearest_object(agent_idx=0, objects=targets)
            result, msg = agent_walk_to_single_step(self, target=target)
        elif action == ActionSpace.WALK_TO_NEAREST_CONTAINER:
            containers = self.controller.containers
            target = self.controller.find_nearest_object(agent_idx=0, objects=containers)
            result, msg = agent_walk_to_single_step(self, target=target)
        elif action == ActionSpace.PICK_UP_NEAREST:
            targets = [idx for idx in self.controller.targets if idx not in self.controller.finished]
            target = self.controller.find_nearest_object(agent_idx=0, objects=targets)
            result, msg = agent_pickup(self, target=target, env_type="wind")
        elif action == ActionSpace.DROP:
            containers = self.controller.containers
            target = self.controller.find_nearest_object(agent_idx=0, objects=containers)
            result, msg = agent_drop(self, container=target, env_type="wind")
        elif action == ActionSpace.EXPLORE:
            target = None
            result, msg = agent_explore(self)
            reward -= 0.5
        elif action == ActionSpace.WALK_TO_RANDOM_OBJECT_IN_SIGHT:
            # if self.last_action == ActionSpace.WALK_TO_RANDOM_OBJECT_IN_SIGHT:
            #     target = self.last_target
            # else:
            #     obs = self.controller._obs()
            #     obj_ids = np.unique(obs["sem_map"]["id"])
            #     targets = [self.controller.manager.get_real_id(idx) for idx in obj_ids if self.controller.manager.get_real_id(idx) not in self.controller.finished]
            #     target = int(self.RNG.choice(targets)) if len(targets) > 0 else None
            target = None
            if target is None:
                result, msg = False, "no object in sight"
            else:
                result, msg = agent_walk_to_single_step(self, target=target)
        self.last_action = action
        self.last_target = target
        
        if result == False:
            reward -= 2
        # status, changed_agents = self.controller.next_key_frame()
        obs, info = self.controller._obs(), self.controller._info()
        info['message'] = msg
        info['success'] = result
        
        # if status[agent_idx] == ActionStatus.success:
        #     obs['status'] = 0
        # elif status[agent_idx] == ActionStatus.ongoing:
        #     obs['status'] = 1
        # else:
        #     obs['status'] = 2
        
        reward += self.controller._reward()
        done = self.controller._done()
        self.num_step += 1
        if self.num_step >= self.max_step:
            done = True
        
        self.done = done
        info['action'] = action
        info['reward'] = reward
        return obs["RL"], reward, done, info
    
    def seed(self, seed):
        self.RNG = np.random.RandomState(seed)
    
    """
    better not use it along with self.step(), it messes with last_action and last_target
    """
    def get_challenge_action(self, action):
        target = None
        ret = None
        if action == ActionSpace.WALK_TO_NEAREST_TARGET:
            targets = [idx for idx in self.controller.targets if idx not in self.controller.finished]
            target = self.controller.find_nearest_object(agent_idx=0, objects=targets)
            ret = "walk_to_single", self.controller.manager.get_renumbered_id(target)
        elif action == ActionSpace.WALK_TO_NEAREST_CONTAINER:
            containers = self.controller.containers
            target = self.controller.find_nearest_object(agent_idx=0, objects=containers)
            ret = "walk_to_single", self.controller.manager.get_renumbered_id(target)
        elif action == ActionSpace.PICK_UP_NEAREST:
            targets = [idx for idx in self.controller.targets if idx not in self.controller.finished]
            target = self.controller.find_nearest_object(agent_idx=0, objects=targets)
            ret = "pick_up", self.controller.manager.get_renumbered_id(target)
        elif action == ActionSpace.DROP:
            containers = self.controller.containers
            target = self.controller.find_nearest_object(agent_idx=0, objects=containers)
            ret = "drop", self.controller.manager.get_renumbered_id(target)
        elif action == ActionSpace.EXPLORE:
            target = None
            ret = "explore", None
        elif action == ActionSpace.WALK_TO_RANDOM_OBJECT_IN_SIGHT:
            try:
                if self.last_action == ActionSpace.WALK_TO_RANDOM_OBJECT_IN_SIGHT:
                    target = self.last_target
                else:
                    obs = self.controller._obs()
                    obj_ids = np.unique(obs["sem_map"]["id"])
                    targets = [self.controller.manager.get_real_id(idx) for idx in obj_ids if self.controller.manager.get_real_id(idx) not in self.controller.finished]
                    target = int(self.RNG.choice(targets)) if len(targets) > 0 else None
            except:
                target = None
            if target is None:
                ret = "explore", None
            else:
                ret = "walk_to_single", self.controller.manager.get_renumbered_id(target)
        self.last_action = action
        self.last_target = target
        return ret