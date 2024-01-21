import random
class RuleBasedAgent:
    def __init__(self,
                 task,
                 ):
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
        self.last_action = None

    def goal2description(self, goal_objects):
        s = "Put "
        for object_name in goal_objects:
            s += f"{object_name}, "
        s = s[:-2] + f" to {self.target_location}."
        return s

    def reset(self, goal_objects, objects_info):
        self.target_objects = goal_objects
        self.goal_desc = self.goal2description(goal_objects)
        self.satisfied = []
        self.nearest_object = []
        self.last_action = None
        print(goal_objects, objects_info)
        
    def get_available_plans(self):
        actions = []
        actions.append(("explore", None))
        if self.last_action == None:
            return None, len(actions), None, actions
        if len(self.holding_objects) == 0 and len(self.nearest_object) > 0:
            action = ("pick_up", int(self.nearest_object[0]['id']))
            actions.append(action)
        if len(self.holding_objects) > 0 and len(self.nearest_object) > 0 and self.task in ['wind']:
            action = ("drop", int(self.nearest_object[0]['id']))
            actions.append(action)
        if len(self.holding_objects) > 0 and self.task in ['fire', 'flood']:
            action = ("drop", None)
            actions.append(action)
        for obj in self.object_list:
            target_flag = False
            for target_object in self.target_objects:
                if target_object == obj['category']:
                    target_flag = True
                    break
            if target_flag and len(self.holding_objects) == 0 and len(self.nearest_object) == 0:
                action = ("walk_to", obj['id'])
                actions.append(action)
        if len(self.holding_objects) == 0 and (self.last_action == None or 'walk_to' not in self.last_action):
            actions.append(("explore", None))
        return None, len(actions), None, actions

    def choose_target(self, state, processed_input):
        if self.debug:
            print(processed_input)
        current_step = processed_input['step']
        holding_objects = processed_input['holding_objects']
        object_list = processed_input['explored_object_name_list']
        nearest_object = processed_input['nearest_object']
        action_result = processed_input['action_result']
        action, info = self.run(current_step, holding_objects, nearest_object, object_list)
        return action

    def run(self, current_step, holding_objects, nearest_object, object_list):
        info = {}
        print("current_step", current_step)
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.nearest_object = nearest_object

        available_plans, num, available_plan_list, actions = self.get_available_plans()
        if num == 0:
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                         "plan": None})
            return plan, info
        print("available_plans", actions)
        if self.last_action is not None and self.last_action in actions and 'explore' not in self.last_action:
            self.last_action = self.last_action
        else: self.last_action = actions[random.randint(0, len(actions) - 1)]
        return self.last_action, info
