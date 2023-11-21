import pdb
import shutil

import os
import json
import time
from envs.flood.flood_gym import FloodEnv
from envs.fire.fire_gym import FireEnv
from envs.wind.wind_gym import WindEnv
from envs.flood.utils import ObjectState as FloodObjectState
from envs.fire.fire_utils import ObjectState as FireObjectState
from policy.env_actions import (agent_walk_to, agent_pickup, agent_drop, agent_explore, visualize_obs,
                                agent_walk_to_single_step, low_level_action)
import logging
from tdw.add_ons.third_person_camera import ThirdPersonCamera

def get_target_description(env):
    return env.controller.target

def get_target_name(env):
    return env.controller.target_names

def get_target_ids(env):
    return env.controller.target_ids

def init_logs(output_dir, name = 'simple_example'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class Challenge:
    def __init__(self, env_name, data_dir, output_dir, logger, launch_build=True, port=1071, screen_size=512,
                 map_size_h=512, map_size_v=512, grid_size=0.1, debug=False, max_steps=1500, use_gt=False,
                 reverse_observation=False, record_only=False, record_with_agents=False, use_dino=False,
                 effect_on_agents=False):
        if env_name == "fire":
            env = FireEnv
            max_steps = 1500 if not record_only else 4500
        elif env_name == "flood":
            env = FloodEnv
            max_steps = 1500
        elif env_name == "wind":
            env = WindEnv
            max_steps = 3000
        else:
            assert False
        self.env_name = env_name
        self.effect_on_agents = effect_on_agents
        if debug:
            self.env = env(launch_build=True, screen_size=screen_size, port=port, use_local_resources=True,
                           map_size_h=map_size_h, map_size_v=map_size_v, grid_size=grid_size,
                           image_capture_path=os.path.join(output_dir, "images"), use_dino=use_dino,
                           log_path = os.path.join(output_dir, "log.txt"), reverse_observation=reverse_observation,
                           check_version=False, use_gt=use_gt, record_only=record_only)
        else:
            self.env = env(launch_build=True, screen_size=screen_size, port=port, use_local_resources=True,
                           map_size_h=map_size_h, map_size_v=map_size_v, grid_size=grid_size,
                           image_capture_path=os.path.join(output_dir, "images") if record_with_agents else None,
                           check_version=False, use_gt=use_gt, use_dino=use_dino,
                           log_path=os.path.join(output_dir, "log.txt"),
                           reverse_observation=reverse_observation, record_only=record_only)
        self.logger = logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.debug = debug
        self.output_dir = output_dir
        self.data_dir = data_dir
        #self.env.reset(data_dir=data_dir)
        self.logger.info("done")
        self.holding_object = []
        self.nearest_object = None
        self.have_finished_list = []
        self.high_value = 5
        self.low_value = 1
        self.max_steps = max_steps

    def reset(self):
        self.holding_object = []
        self.nearest_object = None
        self.have_finished_list = []

    def get_target_info(self, target_list):
        value_dict = json.load(open("scenes/scene_configs/value.json"))
        object_attribute_dict = {}
        for target_category in target_list:
            object_attribute_dict[target_category] = {}
        for target_category in target_list:
            if target_category in value_dict:
                object_attribute_dict[target_category]['value'] = self.high_value if value_dict[target_category] == 1 else self.low_value
            else:
                object_attribute_dict[target_category]['value'] = self.low_value
        if self.env_name == 'fire':
            fireproof_dict = json.load(open("scenes/scene_configs/fire.json"))
            for target_category in target_list:
                if target_category in fireproof_dict:
                    object_attribute_dict[target_category]['fireproof'] = fireproof_dict[target_category]
                else:
                    object_attribute_dict[target_category]['fireproof'] = 0
        elif self.env_name == 'flood':
            waterproof_dict = json.load(open("scenes/scene_configs/fluid.json"))
            for target_category in target_list:
                if target_category in waterproof_dict:
                    object_attribute_dict[target_category]['waterproof'] = waterproof_dict[target_category]
                else:
                    object_attribute_dict[target_category]['waterproof'] = 0
        return object_attribute_dict

    def id_renumbering(self, id):
        return self.env.controller.manager.id_renumbering[id]

    def id_reverse_renumbering(self, reverse_id):
        for id in self.env.controller.manager.id_renumbering:
            if self.env.controller.manager.id_renumbering[id] == reverse_id:
                return id
        return -1

    def process_input(self, state, action_result, action_info):
        processed_input = {}
        explored_sem_id_map = state['sem_map']['explored']*state['sem_map']['id']
        explored_object_id_list = [int(idx) for idx in set(explored_sem_id_map.flatten()) if int(idx) != 0]
        explored_object_id_list = list(set(explored_object_id_list))
        explored_object_name_list = [self.env.controller.manager.segm.names[self.id_reverse_renumbering(idx)]
                                     for idx in explored_object_id_list]
        explored_object_category_list = [self.env.controller.manager.segm.categories[self.id_reverse_renumbering(idx)]
                                         for idx in explored_object_id_list]
        id_map = set((state['sem_map']['explored'] * state['sem_map']['id']).flatten())
        processed_input['explored_object_name_list'] = [
            {'name': name, 'category': category, 'id': str(idx)} for idx, name, category in zip(explored_object_id_list,
                                                                          explored_object_name_list,
                                                                          explored_object_category_list) if idx in id_map
        ]
        processed_input['holding_objects'] = self.holding_object
        if self.nearest_object != None:
            try:
                processed_input['nearest_object'] = [
                    {'name': self.env.controller.manager.segm.names[self.id_reverse_renumbering(self.nearest_object)],
                     'category': self.env.controller.manager.segm.categories[
                         self.id_reverse_renumbering(self.nearest_object)],
                     'id': str(self.nearest_object)}]
            except:
                pdb.set_trace()
        else:
            processed_input['nearest_object'] = []
        processed_input['step'] = self.env.controller.frame_count
        processed_input['action_result'] = action_result
        processed_input['action_info'] = action_info
        # visualize_obs(self.env, None, str(self.step_num), "output")
        return processed_input

    def hold_object(self):
        self.holding_object.append({'name': self.env.controller.manager.segm.names[self.id_reverse_renumbering(self.nearest_object)],
                                    'category': self.env.controller.manager.segm.categories[self.id_reverse_renumbering(self.nearest_object)],
                                    'id': str(self.nearest_object)})
        self.nearest_object = None

    def drop_object(self):
        obj_id = int(self.holding_object[0]['id'])
        reversed_id = self.id_reverse_renumbering(obj_id)
        # print(reversed_id)
        if reversed_id in self.target_status:
            # print('ok')
            # Can not pick up again, because this target is finished
            self.target_status[reversed_id] = self.env.controller.frame_count
            self.have_finished_list.append(obj_id)
        else:
            # print('not ok')
            self.nearest_object = obj_id
        self.holding_object = []

    def get_score(self):
        total_score = 0
        max_score = 0
        value_dict = json.load(open("data/meta_data/value.json"))
        self.final_states = dict()
        if self.env_name in ["fire", "flood"]:
            waterproof_dict = json.load(open("scenes/scene_configs/fluid.json"))
            print("finally:", self.target_status)
            for target in self.target_status:
                name = self.env.controller.target_id2name[target]
                if name in value_dict:
                    if value_dict[name] == 1:
                        value = self.high_value
                    else:
                        value = self.low_value
                else:
                    value = self.low_value
                if self.target_status[target]:
                    if self.env_name == "fire":
                        if self.env.controller.manager.objects[target].state == FireObjectState.NORMAL:
                            total_score += value
                        else:
                            total_score += value * 0.5
                        self.final_states[target] = self.env.controller.manager.objects[target].state.value
                    elif self.env_name == "flood":
                        if name in waterproof_dict:
                            waterproof = waterproof_dict[name]
                        else:
                            waterproof = 0
                        if waterproof or \
                                self.env.controller.manager.objects[target].state == FloodObjectState.NORMAL or \
                                self.env.controller.manager.objects[target].state == FloodObjectState.FLOATING:
                            total_score += value
                        else:
                            total_score += value * 0.5
                        self.final_states[target] = self.env.controller.manager.objects[target].state.value
                    else:
                        total_score += value
                        self.final_states[target] = 0
                max_score += value
        else:
            total_score = 0
            max_score = 1
        return total_score, max_score

    def submit(self, agent, logger, eval_episodes):
        total_finish = 0.0
        total_score = 0.0
        total_max_score = 0.0
        total_steps = 0
        num_eval_episodes = eval_episodes

        start = time.time()
        for i in range(num_eval_episodes):
            start_time = time.time()
            if not os.path.exists(os.path.join(self.output_dir, str(i))):
                os.makedirs(os.path.join(self.output_dir, str(i)))
            self.logger.info('Episode: {}/{}'.format(i + 1, num_eval_episodes))
            self.logger.info(f"Resetting Environment ... data is {self.data_dir}")
            self.reset()
            self.env.reset(data_dir=self.data_dir)
        #    camera = ThirdPersonCamera(avatar_id="a", position={"x": -0.9, "y": 2.0, "z": 2.3}, look_at=1194112)
        #    self.env.controller.add_ons.append(camera)
            target_description = get_target_description(self.env)
            target_ids = get_target_ids(self.env)
            self.target_status = {target_id: False for target_id in target_ids}
            print('init:', self.target_status)
            target_info = self.get_target_info(get_target_description(self.env))
            if target_description is not None:
                if agent.agent_type in ['greedy', 'llm', 'llmv2', 'mcts', 'mctsv2', 'human', 'record', 'rl', 'rule',
                                        'random', 'custom']:
                    agent.reset(goal_objects=target_description,
                                objects_info=target_info)
                elif agent.agent_type == 'oracle':
                    agent.reset(goal_objects=target_description,
                                objects_info=target_info,
                                controller=self.env.controller,
                                step_limit=self.max_steps)
                else:
                    raise Exception(f"{agent.agent_type} not available")
            else:
                assert False
                agent.reset(output_dir=os.path.join(self.output_dir, str(i)))
            # for debug

            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")
            local_finish = self.env.done
            done = False
            self.step_num = 0
            local_reward = 0.0
            action_result = False
            action_info = ""
            self.env.controller.communicate([])
            if "demo" in self.output_dir:
                for i in range(1500):
                    self.env.controller.communicate([])
                return

            action_logger = open(os.path.join(self.output_dir, "actions.txt"), "w")

            if agent.agent_type == "record":
                while self.env.controller.frame_count < self.max_steps:
                    self.env.controller.communicate([])
                done = True

            while not done:
                # self.env.controller._done()
                if self.env_name in ["fire", "flood"]:
                   print("Target status:")
                   for target in self.target_status:
                        print(target, self.target_status[target], self.env.controller.manager.objects[target].state, self.env.controller.manager.objects[target].position)
                state = self.env.controller._obs()
                # Suppose agent can not see the finished object
                for finished_id in self.have_finished_list:
                    state['sem_map']['id'][state['sem_map']['id'] == finished_id] = 0
                    state['raw']['seg_mask'][state['raw']['seg_mask'] == finished_id] = 0
                self.step_num += 1
                processed_input = self.process_input(state, action_result, action_info)
                processed_input['save_dir'] = str(os.path.join(self.output_dir, str(i)))
                
                if agent.agent_type == "llm" or agent.agent_type == "llmv2":
                    import json
                    with open(os.path.join(self.output_dir, str(i), f"input{self.env.controller.frame_count}.json"), "w") as f:
                        json.dump(processed_input, f, indent=4)
                    visualize_obs(self.env, state, suffix=str(self.env.controller.frame_count), save_dir=os.path.join(self.output_dir, str(i)))

                if agent.agent_type == "mcts" or agent.agent_type == "mctsv2":
                    visualize_obs(self.env, state, suffix=str(self.env.controller.frame_count),
                                  save_dir=os.path.join(self.output_dir, str(i)))
                
                current_action = agent.choose_target(state, processed_input)
                if isinstance(current_action, int) and agent.agent_type in ["rl", "random"]:
                    current_action = self.env.get_challenge_action(current_action)
                print(current_action)
                print(f"step {self.env.controller.comm_counter} action {current_action}", file=action_logger)

                if agent.agent_type == 'oracle':
                    while self.env.controller.frame_count < self.max_steps:
                        print(self.env.controller.frame_count, self.max_steps)
                        agent.save_info()
                        self.env.controller.communicate([])
                    oracle_plan = agent.search_plan()
                    import json
                    json.dump(oracle_plan, open(os.path.join(self.output_dir, f"action-{str(i)}.json"), "w"))
                elif current_action[0] == "walk_to":
                    if self.env_name in ["fire", "flood"]:
                        action_result, action_info = agent_walk_to(self.env, target=self.id_reverse_renumbering(int(current_action[1])),
                                                               max_steps=100, reset_arms=False, arrived_at=1.25, task=self.env_name,
                                                                   effect_on_agents=self.effect_on_agents)
                    else:
                        action_result, action_info = agent_walk_to(self.env, target=self.id_reverse_renumbering(int(current_action[1])),
                                                               max_steps=100, reset_arms=False, arrived_at=2, task=self.env_name,
                                                                   effect_on_agents=self.effect_on_agents)
                    if action_result:
                        self.nearest_object = int(current_action[1])
                    else:
                        self.nearest_object = None
                elif current_action[0].startswith("low_level"):
                    action = current_action[0].split(".")[-1]
                    kwargs = current_action[1]
                    action_result, action_info = low_level_action(env, effect_on_agents=self.effect_on_agents,
                                                                  task=self.env_name, action=action, **kwargs)
                    if action_result and action == "move_by":
                        self.nearest_object = self.env.controller.find_nearest_object()
                    else:
                        self.nearest_object = None
                elif current_action[0] == "walk_to_single":
                    if self.env_name in ["fire", "flood"]:
                        action_result, action_info = agent_walk_to_single_step(self.env, target=self.id_reverse_renumbering(int(current_action[1])),
                                                               reset_arms=False, arrived_at=1.25, task=self.env_name,
                                                                               effect_on_agents=self.effect_on_agents)
                    else:
                        action_result, action_info = agent_walk_to_single_step(self.env, target=self.id_reverse_renumbering(int(current_action[1])),
                                                               reset_arms=False, arrived_at=2, task=self.env_name,
                                                                               effect_on_agents=self.effect_on_agents)
                    if action_result and action_info == "success":
                        self.nearest_object = int(current_action[1])
                    else:
                        self.nearest_object = None
                elif current_action[0] == "pick_up":
                    if current_action[1] is None:
                        print('WARNING: not specifying object id, will pick up the nearest object')
                        if self.nearest_object == None:
                            action_result, action_info = False, "You need to walk to an object first."
                        else:
                            action_result, action_info = agent_pickup(self.env, self.id_reverse_renumbering(self.nearest_object), env_type=self.env_name)
                    else:
                        self.nearest_object = int(current_action[1])
                        action_result, action_info = agent_pickup(self.env, self.id_reverse_renumbering(self.nearest_object), env_type=self.env_name)
                    if action_result:
                        self.hold_object()
                elif current_action[0] == "drop":
                    if current_action[1] is None and self.env_name == "wind" and self.id_reverse_renumbering(self.nearest_object) in self.env.controller.containers:
                        current_action = ("drop", self.nearest_object)
                    if current_action[1] is None:
                        print('WARNING: not specifying object id, will drop to the ground')
                        action_result, action_info = agent_drop(self.env, env_type=self.env_name)
                    else:
                        action_result, action_info = agent_drop(self.env, self.id_reverse_renumbering(int(current_action[1])), env_type=self.env_name)
                    if action_result:
                        self.drop_object()
                elif current_action[0] == "explore":
                    action_result, action_info = agent_explore(self.env)
                elif current_action[0] == "stop":
                    action_result, action_info = True, "stopped"
                    done = True
                elif current_action[0] == "record":
                    while self.env.controller.frame_count < self.max_steps:
                        self.env.controller.communicate([])
                else:
                    assert False, f"action {current_action} not available"
                local_finish = "success" if action_result else f"fail, because {action_info}"
                self.logger.info(
                    f"Executing step {self.step_num} for episode: {i}, action: {current_action}, finish: {local_finish}")

                if self.env.controller.frame_count >= self.max_steps:
                    done = True
                print(self.env.controller.frame_count, self.max_steps)
                have_target_left = False
                for target in self.target_status:
                    if not self.target_status[target]:
                        have_target_left = True
                if not have_target_left:
                    done = True
                if done:
                    break
            action_logger.close()
            score, max_score = self.get_score()
            total_score += score
            total_max_score += max_score
            step = self.env.controller.frame_count
            total_steps += step
            if os.path.isfile(os.path.join(self.output_dir, "log.txt")):
                shutil.move(os.path.join(self.output_dir, "log.txt"), os.path.join(self.output_dir, f"log-{str(i)}.txt"))
            # with open(os.path.join(self.output_dir, str(i), 'result_episode.json'), 'w') as f:
            #     json.dump(result, f)
        avg_score = total_score / num_eval_episodes
        avg_steps = total_steps / num_eval_episodes
        avg_max_score = total_max_score / num_eval_episodes
        results = {
            "avg_score": avg_score,
            "avg_steps": avg_steps,
            "avg_max_score": avg_max_score,
            "total": num_eval_episodes,
            "target_status": self.target_status,
            "final_states": self.final_states,
        }
        import json
        with open(os.path.join(self.output_dir, 'eval_result.json'), 'w') as f:
            json.dump(results, f)
        self.logger.info(f'eval done, avg score {avg_score}, max score {avg_max_score}, avg steps {avg_steps}')
        # self.logger.info('eval done, avg reward {}, avg_finish {}'.format(avg_reward, avg_finish))
        self.logger.info('time: {}'.format(time.time() - start))
        return avg_score / avg_max_score, avg_steps

    def close(self):
        self.env.close()
