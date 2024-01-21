import math
import pdb
import sys
import os
import numpy as np
import json
from envs.flood.utils import ObjectState as FloodObjectState
from envs.fire.fire_utils import ObjectState as FireObjectState
from tdw.add_ons.occupancy_map import OccupancyMap

PATH = os.path.dirname(os.path.abspath(__file__))
# go to parent directory until the folder name is HAZARD
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)
sys.path.append(PATH)
sys.path.append(os.path.join(PATH, "ppo"))

class OracleAgent:
    def __init__(self, task):
        self.task = task
        self.agent_speed = 1.0 / 62
        self.agent_type = "oracle"
        self.goal_objects = None
        self.objects_info = None
        self.controller = None
        self.map_list = []
        self.target_damaged_status_sequence = []
        self.target_position_sequence = []
        self.first_save = True
        self.agent_position = []
        self.step_limit = 0
        self.frame_bias = 0

    def reset(self, goal_objects, objects_info, controller, step_limit):
        self.goal_objects = goal_objects
        self.objects_info = objects_info
        self.controller = controller
        self.first_save = True
        self.step_limit = step_limit
        self.target_damaged_status_sequence = []
        self.target_position_sequence = []
        self.map_list = []
        self.step_limit = 0
        self.frame_bias = 0

    def find_path(self, agent_pos, target, start_step):
        meet = False
        additional_steps = 0
        while not meet:
            additional_steps += 1
            if start_step + additional_steps >= self.step_limit:
                return agent_pos, self.step_limit - start_step, 0
            target_id = self.controller.target_ids.index(target)
            cur_step = max(0, start_step + additional_steps - self.frame_bias)
            try:
                if target_id not in self.target_position_sequence[cur_step]:
                    return agent_pos, 0, 0
                target_position = self.target_position_sequence[cur_step][target_id]
            except Exception:
                pdb.set_trace()
            distance = (agent_pos[0] - target_position[0]) ** 2 + (agent_pos[2] - target_position[2]) ** 2
            distance = math.sqrt(distance)
            if additional_steps * self.agent_speed >= distance:
                meet = True
                agent_pos = self.target_position_sequence[cur_step][target_id]
                value_dict = json.load(open("data/meta_data/value.json"))
                name = self.controller.target_id2name[target]
                if name in value_dict:
                    if value_dict[name] == 1:
                        value = 5
                    else:
                        value = 1
                else:
                    value = 1
                if self.target_damaged_status_sequence[cur_step][target_id]:
                    value /= 2

        return agent_pos, additional_steps, value

    def search_step(self, search_order, agent_pos, step, value):
        if len(search_order) == len(self.controller.target_ids) or step >= self.step_limit:
            return step, search_order, value
        min_step = 1e5
        max_value = -1
        best_order = None
        for idx in self.controller.target_ids:
            if idx not in search_order:
                new_agent_pos, new_step, new_value = self.find_path(agent_pos, idx, step)
                sub_min_step, sub_best_order, sub_best_value = self.search_step(search_order + [idx],
                                                                                new_agent_pos, step + new_step,
                                                                                value + new_value)
                if sub_best_value > max_value or (sub_best_value == max_value and sub_min_step < min_step):
                    max_value = value
                    best_order = sub_best_order
                    min_step = sub_min_step
        return min_step, best_order, max_value

    def search_plan(self):
        print(len(self.controller.target_ids))
        if len(self.controller.target_ids) > 11:
            return []
        self.frame_bias = self.step_limit - len(self.target_position_sequence)
        min_step, best_order, best_value = self.search_step([], self.agent_position, 0, 0)
        print("End search", min_step, best_order, best_value)
        return [("walk_to", idx) for idx in best_order]

    def save_info(self):
        if self.first_save:
            self.agent_position = self.controller.agents[0].dynamic.transform.position
            self.first_save = False
        position = {}
        damaged_status = []
        for idx in self.controller.target_ids:
            obj_status = self.controller.manager.objects[idx]
            position[idx] = obj_status.position
            if self.task == "fire":
                damaged_status.append(obj_status.state == FloodObjectState.FLOODED or
                                      obj_status.state == FloodObjectState.FLOODED_FLOATING)
            elif self.task == "flood":
                damaged_status.append(obj_status.state == FireObjectState.BURNING or
                                      obj_status.state == FireObjectState.BURNT)
            else:
                damaged_status.append(False)
        self.target_damaged_status_sequence.append(damaged_status)
        self.target_position_sequence.append(position)

    def choose_target(self, state, processed_input):
        return "explore", None


if __name__ == "__main__":
    agent = OracleAgent("fire")
