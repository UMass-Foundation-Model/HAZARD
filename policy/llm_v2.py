import numpy as np
import pandas as pd
import inflect
from policy.llm import LLM

class LLMv2(LLM):
    def __init__(self, reasoner_prompt_path = "llm_configs/prompt_v2.csv", **kwargs):
        super().__init__(**kwargs)
        self.env_change_record = {}
        self.last_seen_object = {}
        df = pd.read_csv(reasoner_prompt_path)
        self.reasoner_prompt_template = df['prompt'][0]
        self.inflect_engine = inflect.engine()

    def reset(self, goal_objects, objects_info):
        self.target_objects = goal_objects
        self.objects_info = objects_info
        self.env_change_record = {}
        self.action_history = []
        self.action_history_result = []
        self.env_change_record = {}
        self.last_seen_object = {}

    def save_object_history_info(self, obj_id):
        if self.task in ["fire", "flood"]:
            obj_mask = (self.current_state["raw"]["seg_mask"] == obj_id)
            if type(obj_mask) != np.ndarray:
                temp = self.current_state["raw"]["log_temp"] * obj_mask.cpu().numpy()
                avg_temp = (temp.sum() / obj_mask.sum()).item()
            else:
                temp = self.current_state["raw"]["log_temp"] * obj_mask
                avg_temp = temp.sum() / obj_mask.sum()
            if obj_id not in self.env_change_record:
                self.env_change_record[obj_id] = [avg_temp]
            else:
                self.env_change_record[obj_id].append(avg_temp)
        else:
            id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
            object_points = (id_map == obj_id).nonzero()
            if type(object_points[0]) == np.ndarray:
                object_center = (object_points[0].astype(float).mean(), object_points[1].astype(float).mean())
            else:
                object_center = (object_points[:, 0].float().mean().item(), object_points[:, 1].float().mean().item())
            if obj_id not in self.env_change_record:
                self.env_change_record[obj_id] = [object_center]
            else:
                self.env_change_record[obj_id].append(object_center)

    def get_history_description(self, obj_id):
        steps = self.last_seen_object[obj_id]
        record = self.env_change_record[obj_id]
        id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
        object_points = (id_map == obj_id).nonzero()
        if type(object_points[0]) == np.ndarray:
            center = (object_points[0].astype(float).mean(),
                      object_points[1].astype(float).mean())
        else:
            center = (object_points[:, 0].float().mean().item(),
                      object_points[:, 1].float().mean().item())
        distance = np.array((self.agent_pos[0] - center[0], self.agent_pos[1] - center[1]))
        distance = np.linalg.norm(distance)
        dist_description = f"{round(distance, 2)}"
        if self.task == "fire":
            record_str = ", ".join([f"{str(round(np.exp(temp), 2))} Celsius at step {str(i)}" for
                                    i, temp in zip(steps, record)])
            record_str = f"object location: x {str(round(center[0], 2))}, y {str(round(center[1], 2))}, " \
                         f"object distance from me is {dist_description} m, " \
                         f"object temperature is {record_str}."
            return record_str
        elif self.task == "flood":
            record_str = ", ".join([f"{str(round(temp, 2))} m at step {str(i)}" for i, temp in zip(steps, record)])
            record_str = f"object location: x {str(round(center[0], 2))}, y {str(round(center[1], 2))}, " \
                         f"object distance from me is {dist_description} m, " \
                         f"water level at this object is {record_str}."
            return record_str
        else:
            record_str = f"object distance from me is {dist_description} m, "
            record_str += f"object location is {', '.join([f'[{str(round(center[0], 2))}, {str(round(center[1], 2))} ] at step {str(i)}' for i, temp in zip(steps, record)])}."
            record_str = record_str
            return record_str

    def get_current_observation(self):
        text_list = []
        for obj in self.object_list:
            obj_id = int(obj['id'])
            obj_name = obj['category']
            if obj_id not in self.env_change_record:
                text_list.append("Not seen yet.")
                continue
            id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
            object_points = (id_map == obj_id).nonzero()
            if type(object_points[0]) == np.ndarray:
                center = (object_points[0].astype(float).mean(),
                          object_points[1].astype(float).mean())
            else:
                center = (object_points[:, 0].float().mean().item(),
                          object_points[:, 1].float().mean().item())
            record = self.env_change_record[obj_id][-1]
            step = self.last_seen_object[obj_id][-1]
            if self.task == "fire":
                record_str = f"object {obj_name}, id is {str(obj_id)}, last seen object location: x {str(round(center[0], 2))}, " \
                             f"y {str(round(center[1], 2))}, temperature at step {str(step)} is " \
                             f"{str(round(np.exp(record), 2))} Celsius."
            elif self.task == "flood":
                record_str = f"object {obj_name}, id is {str(obj_id)}, last seen object location: x {str(round(center[0], 2))}, " \
                             f"y {str(round(center[1], 2))}, water level at this object at step {str(step)} is " \
                             f"{str(record)} m."
            else:
                record_str = f"object {obj_name}, id is {str(obj_id)}, object location at this object at step {str(step)} is: " \
                             f"x {str(round(center[0], 2))}, y {str(round(center[1], 2))}."
            text_list.append(record_str)
        return "\n".join([f"{chr(ord('A') + i)}. {record_str}" for i, record_str in enumerate(text_list)])

    def get_history_description_list(self):
        current_objects_ids = set(self.current_state["raw"]['seg_mask'].flatten())
        history_list = []
        for obj_id in current_objects_ids:
            if obj_id == 0:
                continue
            if obj_id not in self.last_seen_object:
                self.last_seen_object[obj_id] = [self.current_step]
            else:
                self.last_seen_object[obj_id].append(self.current_step)
            self.save_object_history_info(obj_id)
        for obj in self.object_list:
            obj_id = int(obj['id'])
            # if obj_id not in self.env_change_record or len(self.env_change_record[obj_id]) < 2:
            if obj_id not in self.env_change_record:
                continue
            history_list.append((obj_id, self.get_history_description(obj_id)))
        if len(history_list) == 0:
            # skip reasoning step
            self.env_change_desc = [None] * len(self.object_list)
            return [(-1, 'Not available')]
        else:
            return [(history_desc[0], f"Object id {str(history_desc[0])}, {history_desc[1]}")
                            for i, history_desc in enumerate(history_list)]

    def run(self, current_step, holding_objects, nearest_object, object_list, action_history, agent_pos):
        info = {}
        print("current_step", current_step)
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.nearest_object = nearest_object
        self.agent_pos = agent_pos
        history_list = self.get_history_description_list()
        progress_desc = self.progress2text()
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        prompt = self.prompt_template.replace('$STATE$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc + "\n")
        prompt = prompt.replace('$TARGET_OBJECTS$', self.objects_list2text() + "\n")

        available_plans, num, available_plan_list, actions = self.get_available_plans()
        action_target_idx_list = [int(act[1]) for act in actions if act[1] != None]
        assert num > 0

        history_list_selected = []
        for idx, history in history_list:
            if idx == -1:
                history_list_selected.append(history)
            elif idx in action_target_idx_list:
                history_list_selected.append(history)

        prompt = prompt.replace('$OBJECT_HISTORY$', "\n".join(history_list_selected) + "\n")
        # if num == 0:
        # 	print("Warning! No available plans! only explore")
        # 	plan = None
        # 	info.update({"num_available_actions": num,
        # 				 "plan": None})
        # 	return plan, info

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)
        print(prompt)

        if self.cot:
            prompt = prompt + " Let's think step by step."
            if self.debug:
                print(f"cot_prompt:\n{prompt}")
            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
            output = outputs[0]
            self.total_cost += usage
            info['cot_outputs'] = outputs
            info['cot_usage'] = usage
            if self.debug:
                print(f"cot_output:\n{output}")
            chat_prompt = [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": output},
                           {"role": "user",
                            "content": "Answer with only one best next action. So the answer is option"}]
            normal_prompt = prompt + output + ' So the answer is'
            outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
            output = outputs[0]
            self.total_cost += usage
            info['output_usage'] = usage
            if self.debug:
                print(f"base_output:\n{output}")
                print(f"total cost: {self.total_cost}")
        else:
            if self.debug:
                print(f"base_prompt:\n{prompt}")
            outputs, usage = self.generator([{"role": "user", "content": prompt}] if self.chat else prompt,
                                            self.sampling_params)
            output = outputs[0]
            info['cot_usage'] = usage
            if self.debug:
                print(f"base_output:\n{output}")
        plan = self.parse_answer(available_plan_list, output)
        self.update_history(plan)
        action = actions[available_plan_list.index(plan)]
        if self.debug:
            print(f"plan: {plan}\n")
        info.update({"num_available_actions": num,
                     "prompts": prompt,
                     "outputs": outputs,
                     "plan": plan,
                     "total_cost": self.total_cost})
        return action, info
