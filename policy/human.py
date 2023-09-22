import pdb
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

class SamplingParameters:
    def __init__(self, debug=False, max_tokens=64, t=0.7, top_p=1.0, n=1, logprobs=1, echo=False):
        self.debug = debug
        self.max_tokens = max_tokens
        self.t = t
        self.top_p = top_p
        self.n = n
        self.logprobs = logprobs
        self.echo = echo

class HumanAgent:
    def __init__(self, prompt_template_path, task,):
        self.rooms_explored = None
        self.goal_desc = None
        self.agent_type = "human"
        self.agent_name = "Bob"
        self.prompt_template_path = prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        self.prompt_template = df['prompt'][0].replace("$AGENT_NAME$", self.agent_name)
        self.generator_prompt_template = None
        self.task = task
        assert task in ['fire', 'flood', 'wind']
        self.target_location = "my bag" if task in ['fire', 'flood'] else "a shopping cart"
        self.object_list = None
        self.holding_objects = None
        self.action_history = []
        self.action_history_result = []
        color_file = open("llm_configs/colors.txt")
        colors = color_file.readlines()
        colors = [color.strip()[1:-1] for color in colors]
        self.colors = [[int(c) for c in color.split(",")] for color in colors]
        self.font = ImageFont.truetype(font='llm_configs/GEMELLI.TTF',
                                  size=np.floor(1.5e-2 * 512 + 5).astype('int32'))

    def update_history(self, action):
        self.action_history.append(action)

    def update_history_action_result(self, result):
        if len(self.action_history) == 0:
            return
        self.action_history_result.append(result)

    def reset(self, goal_objects, objects_info):
        self.target_objects = goal_objects
        self.objects_info = objects_info
        self.goal_desc = self.goal2description(goal_objects)

    def goal2description(self, goal_objects):
        s = "Put "
        for object_name in goal_objects:
            s += f"{object_name}, "
        s = s[:-2] + f" to {self.target_location}."
        return s

    def progress2text(self, current_step, satisfied):
        s = f"I've taken {current_step}/3000 steps. "

        s += f"I've already put "
        unique_satisfied = []
        for x in satisfied:
            if x not in unique_satisfied:
                unique_satisfied.append(x)
        if len(satisfied) == 0:
            s += 'nothing'
        s += ', '.join([f"<{x['category']}> ({x['id']})" for x in unique_satisfied])
        # s += ' to the bed. '
        s += '. '

        assert len(self.holding_objects) < 2
        if len(self.holding_objects) == 1:
            obj = self.holding_objects[0]
            s_hold = f"a target object <{obj['category']}> ({obj['id']}). "
        else:
            s_hold = f"nothing. "

        s += f"I'm holding {s_hold}"

        return s

    def action2text(self, action, object=None, additional_info = None):
        if action[0] == "pick_up":
            text = "pick up the object I walked to or just dropped"
        elif action[0] == "drop":
            assert object != None
            if self.task in ['fire', 'flood']:
                text = f"put <{object['category']}> ({object['id']}) I'm holding in my bag."
            else:
                text = f"put down <{object['category']}> ({object['id']}) I'm holding."
        elif action[0] == "walk_to":
            assert object != None
            if additional_info != None:
                text = f"go to object <{object['category']}> ({object['id']}), [{additional_info}]"
            else:
                text = f"go to object <{object['category']}> ({object['id']})"
        elif action[0] == "explore":
            text = f"turn my head to look around"
        else:
            assert False
        return text

    def get_object_location_description(self):
        object_location_description_list = []
        id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
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
        for center in object_centers:
            distance = np.array((self.agent_pos[0] - center[0], self.agent_pos[1] - center[1]))
            distance = np.linalg.norm(distance)
            description = f"distance from me: {round(distance, 2)}"
            object_location_description_list.append(description)
        return object_location_description_list

    def get_available_plans(self):
        available_plans = []
        actions = []
        if len(self.holding_objects) == 0 and len(self.nearest_object) > 0:
            action = ("pick_up", None)
            available_plans.append(self.action2text(action))
            actions.append(action)
        elif len(self.holding_objects) > 0:
            action = ("drop", None)
            available_plans.append(self.action2text(action, self.holding_objects[0]))
            actions.append(action)
        object_location_description_list = self.get_object_location_description()
        for obj, desc in zip(self.object_list, object_location_description_list):
            action = ("walk_to", obj['id'])
            available_plans.append(self.action2text(action, obj, additional_info=desc))
            actions.append(action)

        action = ("explore", None)
        available_plans.append(self.action2text(action))
        actions.append(action)
        
        action = ("stop", None)
        available_plans.append("stop simulation if you feel like nothing can be done anymore")
        actions.append(action)
        
        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans, actions

    def choose_target(self, state, processed_input):
        agent_map = state["goal_map"]
        agent_points = (agent_map == -2).nonzero()
        if type(agent_points[0]) == np.ndarray:
            agent_pos = (agent_points[0].astype(float).mean(),
                         agent_points[1].astype(float).mean())
        else:
            agent_pos = (agent_points[:, 0].float().mean().item(),
                         agent_points[:, 1].float().mean().item())
        self.current_step = processed_input['step']
        satisfied = [] # TODO
        holding_objects = processed_input['holding_objects']
        object_list = processed_input['explored_object_name_list']
        self.current_state = state
        nearest_object = processed_input['nearest_object']
        action_result = processed_input['action_result']
        self.update_history_action_result(action_result)
        action_history = [f"{action} (success)" if result else f"{action} (fail)" for action, result in
                          zip(self.action_history, self.action_history_result)]
        action = self.run(self.current_step, holding_objects, satisfied, nearest_object, object_list,
                                action_history, agent_pos)
        return action

    def get_answer(self, available_plan_list, prompt):
        st.write("\n\n")
        # done = False
        st.text(prompt)
        st.text("\n".join([f"{str(i)}. {plan}" for i, plan in enumerate(available_plan_list)]))
        while True:
            inp = input("Select plan id")
            try:
                inp = int(inp)
            except:
                continue
            if isinstance(inp, int) and int(inp) >= 0 and int(inp) < len(available_plan_list):
                break
        return int(inp)

    def rgb_to_hex(self, rgb):
        r, g, b = rgb
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    def visualize_obs(self):
        id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
        original_size = id_map.shape[0]
        id_list = set(id_map.flatten())
        image = Image.fromarray((self.current_state['raw']['rgb'] * 255).astype(np.uint8).transpose(1,2,0), "RGB")
        draw = ImageDraw.Draw(image)
        id_map_img = np.zeros((id_map.shape[0], id_map.shape[1], 3))
        colors = random.sample(self.colors, len(id_list))
        rgb_sem = self.current_state['raw']['seg_mask']
        for color, idx in zip(colors, id_list):
            object_points = (rgb_sem == idx).nonzero()
            if len(object_points[0]) == 0:
                continue
            if type(object_points[0]) == np.ndarray:
                object_box = [object_points[1].astype(float).min(),
                              object_points[0].astype(float).min(),
                              object_points[1].astype(float).max(),
                              object_points[0].astype(float).max()]
            else:
                object_box = [object_points[:, 1].float().min().item(),
                              object_points[:, 0].float().min().item(),
                              object_points[:, 1].float().max().item(),
                              object_points[:, 0].float().max().item()]
            draw_text = f"idx: {str(idx)}"
            size = draw.textsize(draw_text, self.font)
            text_origin = np.array([object_box[0], object_box[1] - size[1]])

            draw.rectangle(object_box, outline=self.rgb_to_hex(color), width=2)
            draw.rectangle([tuple(text_origin), tuple(text_origin + size)], fill=self.rgb_to_hex(color))
            draw.text(text_origin, draw_text, fill=self.rgb_to_hex((255, 255, 255)) if sum(color)<750 else \
                self.rgb_to_hex((0, 0, 0)), font=self.font)
            color_image = np.array([color])
            color_image = color_image.repeat(25, axis=0)
            color_image = color_image.reshape(5, 5, -1)
            obj_category = "Unknown"
            for obj in self.object_list:
                if int(obj['id']) == idx:
                    obj_category = obj['category']
            st.write(f"{obj_category} color:")
            st.image(color_image)
            id_map_img[id_map==idx] = color
        st.write("Current map")
        id_map_img = Image.fromarray(np.uint8(id_map_img), 'RGB')
        id_map_img = id_map_img.resize((original_size*3, original_size*3))
        st.image(id_map_img)
        st.write("Image")
        st.image(image)

    def run(self, current_step, holding_objects, satisfied, nearest_object, object_list, action_history, agent_pos):
        info = {}
        print("current_step", current_step)
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.nearest_object = nearest_object
        self.agent_pos = agent_pos
        progress_desc = self.progress2text(current_step, satisfied)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
        prompt = prompt.replace('$PROGRESS$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)
        prompt = prompt.replace('$TARGET_OBJECTS$', ", ".join(self.target_objects))
        prompt = prompt.replace('$TARGET_LOCATION$', self.target_location)

        available_plans, num, available_plan_list, actions = self.get_available_plans()
        if num == 0:
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                         "plan": None})
            return plan

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', "")

        self.visualize_obs()
        plan_idx = self.get_answer(available_plan_list, prompt)
        print(len(actions), plan_idx)
        self.update_history(available_plans[plan_idx])
        action = actions[plan_idx]
        return action
