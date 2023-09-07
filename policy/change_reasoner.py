import pdb
import rule_based
import numpy as np
import openai
import json
import os
import pandas as pd
from openai.error import OpenAIError
import backoff
import inflect
from policy.llm import LLM

class LLMChangeReasoner(LLM):
    def __init__(self, reasoner_prompt_path = "llm/prompt_change_reasoner.csv", **kwargs):
        super().__init__(**kwargs)
        self.env_change_record = {}
        self.last_seen_object = {}
        df = pd.read_csv(reasoner_prompt_path)
        self.reasoner_prompt_template = df['prompt'][0]
        self.inflect_engine = inflect.engine()

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
            # if self.task == "fire":
            #     avg_temp = np.exp(avg_temp)
            #     text = f"object temperature is {str(round(avg_temp, 2))} Celsius"
            # else:
            #     text = f"water level at this object is {str(round(avg_temp, 2))} m"
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
        if self.task == "fire":
            record_str = ", ".join([f"{str(round(np.exp(temp), 2))} Celsius at step {str(i)}" for
                                    i, temp in zip(steps, record)])
            record_str = f"object location: x {str(round(center[0], 2))}, y {str(round(center[1], 2))}, " \
                         f"object temperature is {record_str}."
            return record_str
        elif self.task == "flood":
            record_str = ", ".join([f"{str(round(temp, 2))} m at step {str(i)}" for i, temp in zip(steps, record)])
            record_str = f"object location: x {str(round(center[0], 2))}, y {str(round(center[1], 2))}, " \
                         f"water level at this object is {record_str}."
            return record_str
        else:
            record_str = ", ".join([f"[{str(round(center[0], 2))}, {str(round(center[1], 2))}] at step {str(i)}"
                                    for i, temp in zip(steps, record)])
            record_str = f"object location is {record_str}."
            return record_str

    def get_environment_description_for_reason(self):
        if self.task == "fire":
            return "fire spreading"
        elif self.task == "flood":
            return "flood rising. The flood source comes from the larger side of the first axis, " \
                   "and possibly spread to the whole room"
        else:
            return "object moving because of wild wind"

    def get_format_prompt(self):
        if self.task == "fire":
            return "'A, B, C' if the burning order is A, B, then C"
        elif self.task == "flood":
            return "'A, B, C' if the flooded order is A, B, then C"
        else:
            return "'A: [speed], B: [speed], C: [speed]' by including the moving speed " \
                   "in brackets after each object"

    def get_what_to_predict(self):
        if self.task == "fire":
            return "burning order"
        elif self.task == "flood":
            return "submerged order"
        else:
            return "moving direction and speed"

    def get_example_line(self):
        line_length = len(self.object_list)
        if self.task == "fire":
            return ", ".join([chr(ord('A')+i) for i in range(line_length)])
        elif self.task == "flood":
            return ", ".join([chr(ord('A')+i) for i in range(line_length)])
        else:
            return ", ".join([f"{chr(ord('A')+i)}: [0.0, 0.0]" for i in range(line_length)])

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

    def get_current_object_list_desc(self):
        return ", ".join([chr(ord('A') + i) for i in range(len(self.object_list))])

    def parser_reason_answer(self, output):
        self.env_change_desc = [None] * len(self.object_list)
        if self.task in ['fire', 'flood']:
            output = output.split(", ")
            for order, idx_str in enumerate(output):
                if len(idx_str) > 1:
                    continue
                if ord(idx_str[0]) < ord('A') or ord(idx_str[0])-ord('A') >= len(self.object_list):
                    continue
                text = f"this object may be flooded {self.inflect_engine.ordinal(order + 1)} among all objects" \
                    if self.task == 'flood' else f"this object may be burned {self.inflect_engine.ordinal(order + 1)} " \
                                                 f"among all objects"
                self.env_change_desc[ord(idx_str[0])-ord('A')] = text
        else:
            output = output.split(", ")
            for order, out_str in enumerate(output):
                if ": " not in out_str or out_str.split(": ") != 2:
                    continue
                idx_str, speed = out_str.split(": ")
                if len(idx_str) > 1:
                    continue
                if ord(idx_str[0]) < ord('A') or ord(idx_str[0])-ord('A') >= len(self.object_list):
                    continue
                text = f"this object is moving to direction {speed}"
                self.env_change_desc[ord(idx_str[0])-ord('A')] = text

    def reason_environment_change(self):
        info = {}
        current_objects_ids = set(self.current_state["raw"]['seg_mask'].flatten())
        prompt = self.reasoner_prompt_template.replace("$ENV_DESCRIPTION$",
                                                       self.get_environment_description_for_reason())
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
            if obj_id not in self.env_change_record or len(self.env_change_record[obj_id]) < 2:
                continue
            history_list.append((obj_id, self.get_history_description(obj_id)))
        if len(history_list) == 0:
            # skip reasoning step
            self.env_change_desc = [None] * len(self.object_list)
            return
        else:
            history_list = [f"Object {chr(ord('A')+i)}: id {str(history_desc[0])}, {history_desc[1]}"
                            for i, history_desc in enumerate(history_list)]
        prompt = prompt.replace("$HISTORY_STATES$", "\n".join(history_list))
        prompt = prompt.replace("$CURRENT_OBSERVATION$", self.get_current_observation())
        prompt = prompt.replace("$OBJECT_LIST$", self.get_current_object_list_desc())
        prompt = prompt.replace("$FORMAT_PROMPT$", self.get_format_prompt())
        prompt = prompt.replace("$WHAT_TO_PREDICT$", self.get_what_to_predict())
        prompt = prompt.replace("$EXAMPLE_LINE$", self.get_example_line())
        if self.cot:
            prompt = prompt + " Let's think step by step."
            if self.debug:
                print(f"cot_prompt:\n{prompt}")
            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
            output = outputs[0]
            self.total_cost += usage
            info['reason_cot_outputs'] = outputs
            info['reason_cot_usage'] = usage
            if self.debug:
                print(f"cot_output:\n{output}")
            chat_prompt = [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": output},
                           {"role": "user",
                            "content": "Answer in the format of the given example answer line, so the answer is:"}]
            normal_prompt = prompt + output + ' So the answer is'
            outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
            output = outputs[0]
            self.total_cost += usage
            info['reason_output_usage'] = usage
            if self.debug:
                print(f"reason_base_output:\n{output}")
                print(f"reason total cost: {self.total_cost}")
        else:
            if self.debug:
                print(f"reason_base_prompt:\n{prompt}")
            outputs, usage = self.generator([{"role": "user", "content": prompt}] if self.chat else prompt,
                                            self.sampling_params)
            output = outputs[0]
            info['reason_cot_usage'] = usage
            if self.debug:
                print(f"base_output:\n{output}")
        self.parser_reason_answer(output)

    def get_available_plans(self):
        self.reason_environment_change()
        available_plans = []
        actions = []
        if len(self.holding_objects) == 0 and len(self.nearest_object) > 0:
            action = ("pick_up", self.nearest_object[0]['id'])
            available_plans.append(self.action2text(action, self.nearest_object[0]))
            actions.append(action)
        elif len(self.holding_objects) > 0:
            action = ("drop", None)
            available_plans.append(self.action2text(action, self.holding_objects[0]))
            actions.append(action)
        object_location_description_list = self.get_object_location_description()
        for obj, loc_desc, change_desc in zip(self.object_list, object_location_description_list, self.env_change_desc):
            action = ("walk_to", obj['id'])
            if change_desc != None:
                desc = ", ".join([loc_desc, change_desc])
            else:
                desc = loc_desc
            available_plans.append(self.action2text(action, obj, additional_info=desc))
            actions.append(action)

        action = ("explore", None)
        available_plans.append(self.action2text(action))
        actions.append(action)

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans, actions
