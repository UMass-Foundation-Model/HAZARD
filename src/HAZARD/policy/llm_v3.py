import numpy as np
import pandas as pd
import inflect
from policy.llm_v2 import LLMv2

class LLMv3(LLMv2):
    def __init__(self, reasoner_prompt_path = "policy/llm_configs/prompt_v2.csv", **kwargs):
        super().__init__(**kwargs)
        self.env_change_record = {}
        self.last_seen_object = {}
        df = pd.read_csv(reasoner_prompt_path)
        self.reasoner_prompt_template = df['prompt'][0]
        self.inflect_engine = inflect.engine()

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
