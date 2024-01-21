import pdb
import tiktoken
import openai
import json
import os
import pandas as pd
import numpy as np
import random
from openai import OpenAIError
import backoff
import time


class SamplingParameters:
    def __init__(self, debug=False, max_tokens=64, t=0.7, top_p=1.0, n=1, logprobs=1, echo=False):
        self.debug = debug
        self.max_tokens = max_tokens
        self.t = t
        self.top_p = top_p
        self.n = n
        self.logprobs = logprobs
        self.echo = echo


class LLM:
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
                 lm_id,
                 prompt_template_path,
                 cot,
                 sampling_parameters,
                 task,
                 api_key,
                 model_and_tokenizer_path="",
                 total_max_tokens=4096
                 ):
        self.env_change_record = None
        if type(api_key) == list:
            self.apikey_list = api_key
        else:
            self.apikey_list = [api_key]
        self.model_and_tokenizer_path = model_and_tokenizer_path
        self.apikey_idx = 0
        openai.api_key = self.apikey_list[self.apikey_idx]
        self.agent_type = "llm"
        self.task = task
        assert task in ['fire', 'flood', 'wind']
        self.debug = sampling_parameters.debug
        self.prompt_template_path = prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        df.set_index('type', inplace=True)
        self.prompt_template = df.loc[self.task, 'prompt']
        self.cot = cot
        self.source = source
        self.lm_id = lm_id
        self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id
        self.total_cost = 0
        self.total_max_tokens = total_max_tokens - sampling_parameters.max_tokens

        if self.source == 'openai':
            tiktoken.get_encoding("cl100k_base")
            self.tokenizer = tiktoken.encoding_for_model(self.lm_id)
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif self.source == 'huggingface':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            assert model_and_tokenizer_path != ""
            # model_name = "meta-llama/Llama-2-7b-chat-hf"
            self.model = AutoModelForCausalLM.from_pretrained(model_and_tokenizer_path, device_map="auto",
                                                              load_in_4bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizer_path, add_special_tokens=False,
                                                           use_fast=True)
            self.sampling_params = {
                "max_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "n": sampling_parameters.n,
                "logprobs": sampling_parameters.logprobs,
                "echo": sampling_parameters.echo,
            }
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id):
            @backoff.on_exception(backoff.expo, OpenAIError)
            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    base_prompt = prompt[0]["content"]
                    tokens = self.tokenizer.encode(base_prompt)
                    while len(tokens) >= self.total_max_tokens:
                        base_prompt = self.cut_prompt(base_prompt)
                        tokens = self.tokenizer.encode(base_prompt)
                    prompt[0]["content"] = base_prompt
                    try:
                        if self.chat:
                            response = openai.ChatCompletion.create(
                                model=lm_id, messages=prompt, **sampling_params
                            )
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"llm_configs/chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['message']['content'] for i in
                                                 range(sampling_params['n'])]
                            if 'gpt-4' in self.lm_id:
                                usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage'][
                                    'completion_tokens'] * 0.06 / 1000
                            elif 'gpt-3.5' in self.lm_id:
                                usage = response['usage']['total_tokens'] * 0.002 / 1000
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 				  range(sampling_params['n'])]
                        elif "text-" in lm_id:
                            response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"output/raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 			  range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except OpenAIError as e:
                        print(e)
                        print("Switch key and retry...")
                        self.sleep()
                        raise e
                elif source == 'huggingface':
                    if self.chat:
                        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
                        model_inputs = self.tokenizer(prompt, return_tensors="pt")
                        while model_inputs.input_ids.shape[1] >= self.total_max_tokens:
                            prompt = self.cut_prompt(prompt)
                            model_inputs = self.tokenizer(prompt, return_tensors="pt")
                        model_inputs = model_inputs.to("cuda:0")
                        output = self.model.generate(**model_inputs, max_length=model_inputs.input_ids.shape[1]
                                                     + self.sampling_params["max_tokens"],
                                                     temperature=self.sampling_params['temperature'],
                                                     top_p=self.sampling_params['top_p'])
                        generated_samples = [self.tokenizer.decode(output[0][model_inputs.input_ids.shape[1]:],
                                                                   skip_special_tokens=True)]
                        if self.debug:
                            with open(f"llm_configs/chat_raw.json", 'a') as f:
                                f.write(json.dumps(generated_samples[0], indent=4))
                                f.write('\n')
                    else:
                        raise ValueError("Can not use non-chat huggingface models")
                else:
                    raise ValueError("invalid source")
                # generated_samples = [sample.strip().lower() for sample in generated_samples]
                return generated_samples, usage

            return _generate

        self.generator = lm_engine(self.source, self.lm_id)

        self.object_list = None
        self.holding_objects = None
        self.action_history = []
        self.action_history_result = []
        self.action_info_history = []
        self.agent_pos = None
        self.nearest_object = None
        self.objects_info = None
        self.target_objects = None
        self.current_seen_objects_id = None
        self.current_state = None
        self.current_step = None
        self.save_dir = None

    def sleep(self, sleep_time=0.5):
        self.apikey_idx += 1
        if self.apikey_idx >= len(self.apikey_list):
            self.apikey_idx = 0
        openai.api_key = self.apikey_list[self.apikey_idx]
        time.sleep(sleep_time)

    def cut_prompt_with_given_positions(self, position1, position2, prompt):
        assert position1 in prompt
        prompt = prompt.split(position1)
        if len(prompt) < 2:
            return False, prompt
        head, tail = prompt
        head = head + position1
        assert position2 in tail
        tail = tail.split(position2)
        if len(tail) < 2:
            return False, prompt
        mid, tail = tail
        tail = tail + position2
        mid = mid.split("\n")
        if len(mid) > 0:
            mid = mid[:-1]
            mid = "\n".join(mid)
            prompt = head + mid + tail
            return True, prompt
        return False, prompt

    def cut_prompt(self, prompt):
        for pos1, pos2 in [("Objects states history:\n", "\nAvailable actions:"),
                           ("Target objects:\n", "\nCurrent State:"),
                           ("Previous actions:\n", "\nObjects states history:")]:
            success, prompt = self.cut_prompt_with_given_positions(position1=pos1, position2=pos2, prompt=prompt)
            if success:
                return prompt
        prompt = "\n".join(prompt.split("\n")[:-1])
        return prompt

    def update_history(self, action):
        self.action_history.append(action)

    def update_history_action_result(self, result, info):
        if len(self.action_history) == 0:
            return
        self.action_history_result.append(result)
        self.action_info_history.append(info)

    def reset(self, goal_objects, objects_info):
        self.target_objects = goal_objects
        self.objects_info = objects_info
        self.env_change_record = {}
        self.action_history = []
        self.action_history_result = []

    def objects_list2text(self):
        if self.debug:
            print(self.objects_info)
        s = '\n'.join([
            f"name: {category}, value: {str(self.objects_info[category]['value'])}, attribute: {('waterproof' if self.objects_info[category]['waterproof'] == 1 else 'non-waterproof') if self.task == 'flood' else 'None'}"
            for category in self.objects_info])

        return s

    def parse_answer(self, available_actions, text):
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action in text:
                return action

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(
                    ' ') or f"Option {option}" in text or f"({option})" in text or f"action {option}" in text or (
                    len(text) <= 2 and option in text):
                return action
        print(f"WARNING! Fuzzy match! Text: {text}")
        for i in range(len(available_actions)):
            action = available_actions[i]
            act = "None"
            name = "None"
            id = "None"
            if action.startswith('walk_to'):
                act = 'go to'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('pick_up'):
                act = 'pick'
            elif action.startswith('drop'):
                act = 'put'
            elif action.startswith('explore'):
                act = 'turn'
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action
        if len(text) == 1:
            i = ord(text) - ord('A')
            if i in range(len(available_actions)):
                return available_actions[i]
        print("WARNING! No available action parsed!!! Random choose one")
        return random.choice(available_actions)

    def progress2text(self):
        # s = f"I've taken {current_step}/3000 steps. "
        ##todo: add temp/height/... as current object state

        object_location_description_list = self.get_object_location_description()
        if self.task == 'wind':
            ps = "Shopping carts already found:\n"
            for obj, desc in zip(self.object_list, object_location_description_list):
                if obj['category'] != 'shopping cart':
                    continue
                ps += f"name: {obj['category']}, id: {obj['id']}, distance: {desc} m\n"
        else:
            ps = ""
        ps += "Target objects currently seen:\n"
        for obj, desc in zip(self.object_list, object_location_description_list):
            # print(type(obj['id']), type(self.current_seen_objects_id[0]))
            if obj['category'] not in self.target_objects or obj['id'] not in self.current_seen_objects_id:
                continue
            ps += f"name: {obj['category']}, id: {obj['id']}, value: {self.objects_info[obj['category']]['value']}, distance: {desc} m, "
            if self.task == 'fire':
                if obj['id'] not in self.env_change_record:
                    ps += f"temperature: unknown\n"
                else:
                    ps += f"temperature: {str(round(np.exp(self.env_change_record[obj['id']][-1]), 2))} Celsius\n"
            elif self.task == 'flood':
                if obj['id'] not in self.env_change_record:
                    ps += f"water level: unknown\n"
                else:
                    ps += f"water level: {str(round(self.env_change_record[obj['id']][-1], 2))} m\n"
            else:
                ps += f"status: {'Unknown'}\n"
        ps += 'Target objects previously seen:\n'
        for obj, desc in zip(self.object_list, object_location_description_list):
            if obj['category'] not in self.target_objects or obj['id'] in self.current_seen_objects_id:
                continue
            ps += f"name: {obj['category']}, id: {obj['id']}, value: {self.objects_info[obj['category']]['value']}, distance: {desc} m, "
            if self.task == 'fire':
                if obj['id'] not in self.env_change_record:
                    ps += f"temperature: unknown\n"
                else:
                    ps += f"temperature: {str(round(np.exp(self.env_change_record[obj['id']][-1]), 2))} Celsius\n"
            elif self.task == 'flood':
                if obj['id'] not in self.env_change_record:
                    ps += f"water level: unknown\n"
                else:
                    ps += f"water level: {str(round(self.env_change_record[obj['id']][-1], 2))} m\n"
            else:
                ps += f"status: {'Unknown'}\n"
        return ps

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
                object_centers.append((object_points[:, 0].float().mean().item(),
                                       object_points[:, 1].float().mean().item()))
        # TODO use relative location to agent
        for center in object_centers:
            distance = np.array((self.agent_pos[0] - center[0], self.agent_pos[1] - center[1]))
            distance = np.linalg.norm(distance)
            # description = f"distance from me: {round(distance, 2)}"
            description = f"{round(distance, 2)}"
            object_location_description_list.append(description)
        return object_location_description_list

    def update_object_status(self):
        for o_id in self.current_seen_objects_id:
            obj_id = int(o_id)
            if self.task in ["fire", "flood"]:
                obj_mask = (self.current_state["raw"]["seg_mask"] == obj_id)
                # import pdb; pdb.set_trace()
                if type(obj_mask) != np.ndarray:
                    temp = self.current_state["raw"]["log_temp"] * obj_mask.cpu().numpy()
                    avg_temp = (temp.sum() / obj_mask.sum()).item()
                else:
                    temp = self.current_state["raw"]["log_temp"] * obj_mask
                    avg_temp = temp.sum() / obj_mask.sum()
                if obj_id not in self.env_change_record:
                    self.env_change_record[str(obj_id)] = [avg_temp]
                else:
                    self.env_change_record[str(obj_id)].append(avg_temp)
            # if self.task == "fire":
            #     avg_temp = np.exp(avg_temp)
            #     text = f"object temperature is {str(round(avg_temp, 2))} Celsius"
            # else:
            #     text = f"water level at this object is {str(round(avg_temp, 2))} m"
        # else:
        # 	id_map = self.current_state['sem_map']['explored'] * self.current_state['sem_map']['id']
        # 	object_points = (id_map == obj_id).nonzero()
        # 	if type(object_points[0]) == np.ndarray:
        # 		object_center = (object_points[0].astype(float).mean(), object_points[1].astype(float).mean())
        # 	else:
        # 		object_center = (object_points[:, 0].float().mean().item(), object_points[:, 1].float().mean().item())
        # 	if obj_id not in self.env_change_record:
        # 		self.env_change_record[obj_id] = [object_center]
        # 	else:
        # 		self.env_change_record[obj_id].append(object_center)

    def get_available_plans(self):
        available_plans = []
        actions = []
        if len(self.holding_objects) == 0:
            for obj in self.object_list:
                if obj['category'] not in self.target_objects or (
                        len(self.holding_objects) > 0 and obj['id'] == self.holding_objects[0]['id']):
                    continue
                action = ("walk_to", obj['id'])
                available_plans.append(f"go pick up object <{obj['category']}> ({obj['id']})")
                actions.append(action)
        else:
            if self.task == "wind":
                for obj in self.object_list:
                    if obj['category'] != 'shopping cart':
                        continue
                    action = ("walk_to", obj['id'])
                    available_plans.append(f"go put object into <{obj['category']}> ({obj['id']})")
                    actions.append(action)
            else:
                action = ("drop", None)
                available_plans.append(f"put the holding object in my bag")
                actions.append(action)

        if len(self.action_history) == 0 or self.action_history[-1] != 'look around':
            action = ("explore", None)
            available_plans.append("look around")
            actions.append(action)

        if len(actions) == 0:
            for obj in self.object_list:
                action = ("walk_to", obj['id'])
                available_plans.append(f"go to object <{obj['category']}> ({obj['id']})")
                actions.append(action)
        if len(actions) > 10:
            actions = actions[:10]
            available_plans = available_plans[:10]
        if len(actions) == 0:
            action = ("explore", None)
            available_plans.append("look around")
            actions.append(action)
        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans, actions

    def action_result_to_description(self, result, info):
        if result:
            return "success"
        elif info == 'max steps reached':
            return "paused after taking 100 steps"
        else:
            return f"fail, because {info}"

    def choose_target(self, state, processed_input):
        self.save_dir = processed_input['save_dir']
        agent_map = state["goal_map"]
        agent_points = (agent_map == -2).nonzero()
        if type(agent_points[0]) == np.ndarray:
            agent_pos = (agent_points[0].astype(float).mean(), agent_points[1].astype(float).mean())
        else:
            agent_pos = (agent_points[:, 0].float().mean().item(), agent_points[:, 1].float().mean().item())
        self.current_step = processed_input['step']
        holding_objects = processed_input['holding_objects']
        object_list = processed_input['explored_object_name_list']
        self.current_state = state
        self.current_seen_objects_id = [str(x) for x in list(set(self.current_state["raw"]['seg_mask'].flatten()))]
        if self.debug:
            print(f"current seen objects id: {self.current_seen_objects_id}")
        nearest_object = processed_input['nearest_object']
        action_result = processed_input['action_result']
        action_info = processed_input['action_info']
        self.update_object_status()
        if self.task == 'wind':
            if len(processed_input['holding_objects']) == 0 and \
                    len(processed_input['nearest_object']) > 0 and \
                    processed_input['nearest_object'][0]['category'] in self.target_objects:
                return "pick_up", processed_input['nearest_object'][0]['id']
            if len(processed_input['holding_objects']) > 0 and \
                    len(self.action_history) > 0 and \
                    self.action_history[-1].startswith('go put object into <shopping cart>') and \
                    processed_input['action_result']:
                return "drop", self.action_history[-1].split('(')[1].split(')')[0]
        else:
            if len(processed_input['holding_objects']) > 0 and \
                    processed_input['holding_objects'][0]['category'] in self.target_objects:
                return "drop", None
            if len(processed_input['nearest_object']) > 0 and \
                    processed_input['nearest_object'][0]['category'] in self.target_objects:
                return "pick_up", processed_input['nearest_object'][0]['id']

        self.update_history_action_result(action_result, action_info)
        action_history = [f"{action} ({self.action_result_to_description(result, info)})" for action, result, info in
                          zip(self.action_history, self.action_history_result, self.action_info_history)]
        action, info = self.run(self.current_step, holding_objects, nearest_object, object_list, action_history,
                                agent_pos)
        import json
        with open(os.path.join(self.save_dir, f'{self.current_step:04d}_info.json'), 'w') as f:
            json.dump(info, f, indent=4)
        return action

    def run(self, current_step, holding_objects, nearest_object, object_list, action_history, agent_pos):
        info = {}
        print("current_step", current_step)
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.nearest_object = nearest_object
        self.agent_pos = agent_pos
        progress_desc = self.progress2text()
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        prompt = self.prompt_template.replace('$STATE$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc + "\n")
        prompt = prompt.replace('$TARGET_OBJECTS$', self.objects_list2text() + "\n")

        available_plans, num, available_plan_list, actions = self.get_available_plans()
        assert num > 0
        # if num == 0:
        # 	print("Warning! No available plans! only explore")
        # 	plan = None
        # 	info.update({"num_available_actions": num,
        # 				 "plan": None})
        # 	return plan, info

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

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
