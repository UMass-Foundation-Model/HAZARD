import pdb

import numpy as np
import copy
import random

class mcts_state_H:
    def __init__(self, object_info, target_list: list = None, container_list: list = None, agent_pos: list = None,
                 holding_objects: list = None, last_action: tuple = None, last_action_success: bool = True,
                 task = None, last_cost = 10000, status_history = None):
        self.target_list = target_list
        self.container_list = container_list
        self.agent_pos = agent_pos
        self.holding_objects = holding_objects
        self.last_action = last_action
        self.task = task
        self.c = 5e-2 if self.task in ['fire', 'flood'] else 5e-2
        self.status_history = status_history
        if self.last_action is None: self.last_action = ('None', None)
        self.last_action_success = last_action_success
        self.is_expanded = False
        self.num_visited = 0
        self.sum_value = 0
        self.children = None
        self.children_cost = None
        self.last_cost = last_cost
        self.objects_info = object_info
    def __str__(self):
        return f"target_list: {self.target_list}, container_list: {self.container_list}, agent_pos: {self.agent_pos}, holding_objects: {self.holding_objects}, last_action: {self.last_action}, last_action_success: {self.last_action_success}"

    def get_distance(self, pos1, pos2):
        dis = min(np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) * self.c, 10)
        return dis
    
    def check_distance_increase(self, cost):
        if cost > self.last_cost:
            return 1
        else:
            return 0
    
    def get_available_trans(self):
        mcts_trans = []
        miss_target_flag = False

        if len(self.holding_objects) > 0:
            if self.task in ['fire', 'flood']:
                mcts_trans.append({'action': ("drop", None), 'cost': 0}) # TODO: change when in wind setting
            elif self.task == 'wind':
                if 'walk_to' in self.last_action[0] and self.last_action_success == True:
                    mcts_trans.append({'action': ("drop", self.last_action[1]), 'cost': 0})
                else:
                    for obj in self.container_list:
                        if obj['id'] == self.last_action[1] or 'walk_to' not in self.last_action[0]:
                            mcts_trans.append({'action': ("walk_to", obj['id']), 'cost': self.get_distance(self.agent_pos, obj['pos'])})
            else:
                raise NotImplementedError
            
        if len(self.holding_objects) == 0 and 'explore' not in self.last_action[0]:
            if not('walk_to' in self.last_action[0] and self.last_action_success == True):
                mcts_trans.append({'action': ("explore", None), 'cost': 2.5})

        if len(self.holding_objects) == 0 and 'walk_to' in self.last_action[0]:
            if self.last_action_success == False:
                goal_pos = None
                for obj in self.target_list:
                    if obj['id'] == self.last_action[1]:
                        goal_pos = obj['pos']
                        break
                if goal_pos is None:
                    miss_target_flag = True
                else:
                    mcts_trans.append({'action': copy.deepcopy(self.last_action), 'cost': self.get_distance(self.agent_pos, goal_pos) + self.check_distance_increase(self.get_distance(self.agent_pos, goal_pos)) * 5})
            else:
                mcts_trans.append({'action': ("pick_up", self.last_action[1]), 'cost': 0}) # TODO: change when in wind setting
        
        if len(self.holding_objects) == 0 and ('walk_to' not in self.last_action[0] or miss_target_flag):
            for obj in self.target_list:
                mcts_trans.append({'action': ("walk_to", obj['id']), 'cost': self.get_distance(self.agent_pos, obj['pos'])})

        # consider value
        for action in mcts_trans:
            if action['action'][1] == None or action['cost'] == 0:
                continue
            value = 1
            for object in self.target_list:
                if action['action'][1] == object['id']:
                    value = self.objects_info[object['category']]['value']
            if self.task in ['fire', 'flood']:
                vulnerable = 1e5
                for object in self.target_list:
                    if action['action'][1] == object['id']:
                        value = self.objects_info[object['category']]['value']
                        if 'waterproof' in self.objects_info[object['category']]:
                            vulnerable = 1e5 if self.objects_info[object['category']]['waterproof'] == 1 else 0.01
                        elif 'fireproof' in self.objects_info[object['category']]:
                            vulnerable = 1000 if self.objects_info[object['category']]['fireproof'] == 1 else 100
                if int(action['action'][1]) not in self.status_history:
                    pass
                elif len(self.status_history[int(action['action'][1])]) == 1:
                    vulnerable -= self.status_history[int(action['action'][1])][-1]
                else:
                    delta = self.status_history[int(action['action'][1])][-1] - self.status_history[int(action['action'][1])][-2]
                    status_prediction = self.status_history[int(action['action'][1])][-1] + delta
                    vulnerable -= status_prediction
                if vulnerable < 0:
                    value /= 2
            action['cost'] /= value
        return mcts_trans

    def apply_trans(self, action):
        new_state = copy.deepcopy(self)
        new_state.last_action = copy.deepcopy(action)
        new_state.last_action_success = True
        new_state.is_expanded = False
        new_state.num_visited = 0
        new_state.sum_value = 0
        new_state.children = None
        new_state.children_cost = None
        if action[0] == 'drop':
            new_state.holding_objects = []
        if action[0] == 'pick_up':
            new_state.holding_objects = [copy.deepcopy(action[1])]
            for i in range(len(new_state.target_list)):
                if new_state.target_list[i]['id'] in [action[1], int(action[1])]:
                    new_state.target_list.pop(i)
                    break
            else: assert False
        if action[0] == 'walk_to':
            for obj in new_state.target_list:
                if obj['id'] == action[1]:
                    new_state.agent_pos = obj['pos']
                    break
            else:
                for obj in new_state.container_list:
                    if obj['id'] == action[1]:
                        new_state.agent_pos = obj['pos']
                        break
                else: assert False
        return new_state
    
    def final_cost(self):
        remain_dis = 0
        for i in range(len(self.target_list)):
            remain_dis += max(2 * self.get_distance(self.agent_pos, self.target_list[i]['pos']) - 5, 0)
        return remain_dis

class GreedyAgent:
    def __init__(self, task):
        self.debug = False
        self.rooms_explored = None
        self.goal_desc = None
        self.agent_type = "mcts"
        self.agent_name = "Bob"
        self.rooms = []
        self.total_cost = 0
        self.task = task
        assert task in ['fire', 'flood', 'wind']
        self.target_location = "my bag" if task in ['fire', 'flood'] else "a shopping cart"
        self.current_room = None
        self.object_list = None
        self.holding_objects = None
        self.obj_per_room = None
        self.satisfied = None
        self.num_simulation = 2000
        self.c_base = 1000000
        self.c_init = 0.1
        self.discount = 0.95
        self.max_rollout_step = 20
        self.last_cost = 10000
        self.status_history = {}
        #DWH: constant from wah code, maybe change later

    def goal2description(self, goal_objects):
        s = "Put "
        for object_name in goal_objects:
            s += f"{object_name}, "
        s = s[:-2] + f" to {self.target_location}."
        return s

    def reset(self, goal_objects, objects_info):
        self.target_objects = goal_objects
        self.container_objects = ['shopping cart']
        self.goal_desc = self.goal2description(goal_objects)
        self.satisfied = []
        self.nearest_object = []
        self.local_step = 0
        self.curr_state = None
        self.need_to_go = None
        self.last_action = None
        self.last_action_result = True
        self.objects_info = objects_info

    def save_object_history_info(self, state, obj_id):
        if self.task in ["fire", "flood"]:
            obj_mask = (state["raw"]["seg_mask"] == obj_id)
            if type(obj_mask) != np.ndarray:
                temp = state["raw"]["log_temp"] * obj_mask.cpu().numpy()
                avg_temp = (temp.sum() / obj_mask.sum()).item()
            else:
                temp = state["raw"]["log_temp"] * obj_mask
                avg_temp = temp.sum() / obj_mask.sum()
            if obj_id not in self.status_history:
                self.status_history[obj_id] = [avg_temp]
            else:
                self.status_history[obj_id].append(avg_temp)
        else:
            id_map = state['sem_map']['explored'] * state['sem_map']['id']
            object_points = (id_map == obj_id).nonzero()
            if type(object_points[0]) == np.ndarray:
                object_center = (object_points[0].astype(float).mean(), object_points[1].astype(float).mean())
            else:
                object_center = (object_points[:, 0].float().mean().item(), object_points[:, 1].float().mean().item())
            if obj_id not in self.status_history:
                self.status_history[obj_id] = [object_center]
            else:
                self.status_history[obj_id].append(object_center)

    def choose_target(self, state, processed_input):
        current_objects_ids = set(state["raw"]['seg_mask'].flatten())
        for obj_id in current_objects_ids:
            self.save_object_history_info(state, obj_id)
        if self.debug:
            print(processed_input)
        self.object_list = copy.deepcopy(processed_input['explored_object_name_list'])
        print(self.object_list)
        self.holding_objects = processed_input['holding_objects']
        self.last_action_result = processed_input['action_result']
        self.curr_state = state
        agent_map = state["goal_map"]
        agent_points = (agent_map == -2).nonzero()
        if type(agent_points[0]) == np.ndarray:
            agent_pos = (agent_points[0].astype(float).mean(),
                         agent_points[1].astype(float).mean())
        else:
            agent_pos = (agent_points[:, 0].float().mean().item(),
                         agent_points[:, 1].float().mean().item())
        id_map = self.curr_state['sem_map']['explored'] * self.curr_state['sem_map']['id']
        object_ids = [int(obj['id']) for obj in self.object_list]
        object_centers = []
        for idx in object_ids:
            object_points = (id_map == idx).nonzero()
            if type(object_points[0]) == np.ndarray:
                object_centers.append((object_points[0].astype(float).mean(),
                                       object_points[1].astype(float).mean()))
            else:
                object_centers.append((object_points[:,0].float().mean().item(),
                                       object_points[:,1].float().mean().item()))
        self.need_to_go = []
        self.containers = []
        for i in range(len(self.object_list)):
            # ID, CLASS, NAME, POS
            self.object_list[i]['pos'] = object_centers[i]
            target_flag = False
            container_flag = False
            for target_object in self.target_objects:
                if target_object == self.object_list[i]['category']:
                    target_flag = True
                    break
            for container_object in self.container_objects:
                if container_object == self.object_list[i]['category']:
                    container_flag = True
                    break
            if target_flag:
                self.need_to_go.append(self.object_list[i])
            if container_flag:
                self.containers.append(self.object_list[i])

        mcts_root = mcts_state_H(
                object_info=self.objects_info,
                target_list = copy.deepcopy(self.need_to_go), 
                container_list = copy.deepcopy(self.containers), 
                agent_pos = copy.deepcopy(agent_pos), 
                holding_objects = copy.deepcopy(self.holding_objects), 
                last_action = copy.deepcopy(self.last_action), 
                last_action_success = self.last_action_result,
                last_cost = self.last_cost,
                status_history = self.status_history,
                task = self.task,)
        print(mcts_root.get_available_trans())
        
        plan, action = self.mcts(mcts_root)
        if action is None:
            action = ("explore", None)
        print(plan, action)
        self.last_action = action
        return action

    def calculate_score(self, curr_node: mcts_state_H, child: mcts_state_H, expr = True):
        parent_visit_count = curr_node.num_visited
        self_visit_count = child.num_visited

        if self_visit_count == 0:
            u_score = 1e6
            q_score = 0
        else:
            exploration_rate = np.log((1 + parent_visit_count + self.c_base) / self.c_base) + self.c_init
            u_score = exploration_rate * np.sqrt(parent_visit_count) / float(1 + self_visit_count)
            q_score = child.sum_value / self_visit_count

        score = q_score + u_score
        if expr == False: 
            score = q_score
        return score

    def select_child(self, curr_node: mcts_state_H, expr = True):
        scores = [curr_node.children_cost[i] for i in range(len(curr_node.children_cost))]
        costs = curr_node.children_cost
        if len(scores) == 0: return None
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selected_child_index = random.choice(maxIndex)
        selected_child = curr_node.children[selected_child_index]
        return selected_child, costs[selected_child_index], curr_node.get_available_trans()[selected_child_index]['action']

    def initialize_children(self, curr_node: mcts_state_H):
        available_trans = curr_node.get_available_trans()
        n_state = []
        n_cost = []
        for trans in available_trans:
            next_node = curr_node.apply_trans(trans['action'])
            n_state.append(next_node)
            n_cost.append(trans['cost'])
        if len(n_state) == 0: n_state = None
        curr_node.children = n_state
        curr_node.children_cost = n_cost
        return n_state

    def expand(self, leaf_node: mcts_state_H):
        expanded_leaf_node = self.initialize_children(leaf_node)
        if expanded_leaf_node is not None:
            leaf_node.is_expanded = True
        return leaf_node

    def mcts(self, root: mcts_state_H):
        try:
            curr_node = root
            plans = []
            self.expand(curr_node)
            next_node, cost, action = self.select_child(curr_node, expr = False)
        except:
            return [], None
        return plans, action  # return the plan and action