from challenge import Challenge, init_logs
from policy.llm import LLM, SamplingParameters
from policy.llm_v2 import LLMv2
from policy.change_reasoner import LLMChangeReasoner
from policy.rule_based import RuleBasedAgent
from policy.fire_flood_heuristic import GreedyAgent
from policy.human import HumanAgent
from policy.mcts import MCTS
from policy.mctsv2 import MCTSv2
from policy.record_agent import RecordAgent
from policy.rl import RLAgent
from policy.random import RandomAgent
from policy.custom import CustomAgent
from policy.oracal import OracleAgent
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--api_key_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--env_name", type=str, choices=["fire", "flood", "wind"], default="flood")
    parser.add_argument("--agent_name", type=str, choices=["rule", "llm", "llmv2", "mcts",
                                                           "llm+change", "greedy", "human", "mctsv2", "oracle",
                                                           "record", "rl", "random", "custom"], default="llmv2")
    parser.add_argument("--data_dir", type=str, default="data/room_setup_fire/mm_craftroom_2a-1")
    parser.add_argument("--port", type=int, default=1071)
    parser.add_argument("--max_test_episode", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--prompt_path", type=str, default="llm_configs/prompt.csv")
    parser.add_argument("--lm_id", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--lm_source", type=str, choices=['openai', 'huggingface'], default="openai")
    parser.add_argument("--model_and_tokenizer_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--reverse_observation", action='store_true', default=False)
    parser.add_argument("--record_with_agents", action='store_true', default=False)
    parser.add_argument("--grid_size", type=float, default=0.1)
    parser.add_argument("--use_gt", action='store_true', default=False)
    parser.add_argument("--effect_on_agents", action='store_true', default=False)
    parser.add_argument("--use_dino", action='store_true', default=False)
    return parser.parse_args()

def get_agent(args):
    if args.agent_name == "llm":
        sampling_parameters = SamplingParameters(debug=args.debug, max_tokens=args.max_tokens)
        if args.api_key_file != "":
            api_key_file = open(args.api_key_file)
            api_key_list = api_key_file.readlines()
            api_key_list = [api_key.strip() for api_key in api_key_list]
        else:
            api_key_list = args.api_key
        return LLM(source=args.lm_source, lm_id=args.lm_id, prompt_template_path=args.prompt_path, cot=True,
                   sampling_parameters=sampling_parameters, task=args.env_name, api_key=api_key_list,
                   model_and_tokenizer_path=args.model_and_tokenizer_path)
    if args.agent_name == "llmv2":
        sampling_parameters = SamplingParameters(debug=args.debug, max_tokens=args.max_tokens)
        if args.api_key_file != "":
            api_key_file = open(args.api_key_file)
            api_key_list = api_key_file.readlines()
            api_key_list = [api_key.strip() for api_key in api_key_list]
        else:
            api_key_list = args.api_key
        return LLMv2(source=args.lm_source, lm_id=args.lm_id, prompt_template_path=args.prompt_path, cot=True,
                     sampling_parameters=sampling_parameters, task=args.env_name, api_key=api_key_list,
                     model_and_tokenizer_path=args.model_and_tokenizer_path)
    elif args.agent_name == "llm+change":
        sampling_parameters = SamplingParameters(debug=args.debug)
        if args.api_key_file != "":
            api_key_file = open(args.api_key_file)
            api_key_list = api_key_file.readlines()
            api_key_list = [api_key.strip() for api_key in api_key_list]
            return LLMChangeReasoner(source="openai", lm_id=args.lm_id, prompt_template_path=args.prompt_path,
                                     cot=True, sampling_parameters=sampling_parameters, task=args.env_name,
                                     api_key=api_key_list)
        else:
            return LLMChangeReasoner(source="openai", lm_id=args.lm_id, prompt_template_path=args.prompt_path,
                                     cot=True, sampling_parameters=sampling_parameters, task=args.env_name,
                                     api_key=args.api_key)
    elif args.agent_name == 'mcts':
        return MCTS(task=args.env_name)
    elif args.agent_name == 'mctsv2':
        return MCTSv2(task=args.env_name)
    elif args.agent_name == 'rule':
        return RuleBasedAgent(task=args.env_name)
    elif args.agent_name == 'human':
        return HumanAgent(task=args.env_name, prompt_template_path=args.prompt_path)
    elif args.agent_name == "greedy":
        # if args.env_name in ["fire", "flood"]:
        #     return HAgent(task=args.env_name)
        # else:
        #     return HAgentWind_FROM_MCTS(task=args.env_name)
        return GreedyAgent(task=args.env_name)
    elif args.agent_name == "record":
        return RecordAgent(task=args.env_name)
    elif args.agent_name == "oracle":
        return OracleAgent(task=args.env_name)
    elif args.agent_name == "rl":
        return RLAgent(task=args.env_name)
    elif args.agent_name == "random":
        return RandomAgent(task=args.env_name)
    elif args.agent_name == "custom":
        return CustomAgent(task=args.env_name)
    else:
        assert False

if __name__ == "__main__":
    args = get_args()
    print(args.data_dir)
    print(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger = init_logs(output_dir=args.output_dir, name=f"{args.env_name}_{args.agent_name}")
    if args.agent_name == "rl":
        challenge = Challenge(env_name=args.env_name, data_dir=args.data_dir, output_dir=args.output_dir, logger=logger,
                              launch_build=not args.debug, debug=args.debug, port=args.port, screen_size=1024,
                              map_size_h=256, map_size_v=256, grid_size=args.grid_size, use_gt=args.use_gt,
                              reverse_observation=args.reverse_observation, record_only=(args.agent_name == "record"),
                              record_with_agents=args.record_with_agents, use_dino=args.use_dino,
                              effect_on_agents=args.effect_on_agents)
    else:
        challenge = Challenge(env_name=args.env_name, data_dir=args.data_dir, output_dir=args.output_dir, logger=logger,
                              launch_build=not args.debug, debug=args.debug, port=args.port, screen_size=1024,
                              grid_size=args.grid_size, use_gt=args.use_gt, reverse_observation=args.reverse_observation,
                              record_only=(args.agent_name == "record"), record_with_agents=args.record_with_agents,
                              use_dino=args.use_dino, effect_on_agents=args.effect_on_agents)
    agent = get_agent(args)
    if os.path.exists(os.path.join(args.data_dir, "log.txt")): # single episode
        challenge.submit(agent=agent, logger=logger, eval_episodes=1)
    else:
        challenge_list = os.listdir(args.data_dir)
        challenge_list = sorted(challenge_list)
        count = 0
        for task in challenge_list:
            if 'craftroom' not in task and 'kitchen' not in task and 'suburb' not in task:
                continue
            count += 1
            if count > args.max_test_episode:
                break
            now_data_dir = os.path.join(args.data_dir, task)
            now_output_dir = os.path.join(args.output_dir, task)
            if not os.path.exists(now_output_dir):
                os.makedirs(now_output_dir)
            eval_result_path = os.path.join(args.output_dir, task, 'eval_result.json')
            if os.path.exists(eval_result_path):
                print(f"{eval_result_path} exists")
                continue # already evaluated
            challenge.output_dir = now_output_dir
            challenge.data_dir = now_data_dir
            print(now_output_dir)
            challenge.submit(agent=agent, logger=logger, eval_episodes=1)
    if challenge.env.controller is not None:
        challenge.env.controller.communicate({"$type": "terminate"})
        challenge.env.controller.socket.close()
