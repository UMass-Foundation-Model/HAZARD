from src.HAZARD.challenge import Challenge, init_logs
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
from policy.rand import RandomAgent
from policy.custom import CustomAgent
from policy.oracal import OracleAgent
from datetime import datetime
import os


DATA_DIR = {
    "fire": "room_setup_fire",
    "flood": "room_setup_fire",
    "wind": "room_setup_wind"
}


def get_examplar_agent(agent_name="mcts", api_key="", api_key_file="", debug=False, max_tokens=512, lm_source="openai",
                       lm_id="gpt-3.5-turbo", model_and_tokenizer_path="", env_name="fire", prompt_path=""):
    if agent_name == "llm":
        sampling_parameters = SamplingParameters(debug=debug, max_tokens=max_tokens)
        if api_key_file != "":
            api_key_file = open(api_key_file)
            api_key_list = api_key_file.readlines()
            api_key_list = [api_key.strip() for api_key in api_key_list]
        else:
            api_key_list = api_key
        return LLM(source=lm_source, lm_id=lm_id, prompt_template_path=prompt_path, cot=True,
                   sampling_parameters=sampling_parameters, task=env_name, api_key=api_key_list,
                   model_and_tokenizer_path=model_and_tokenizer_path)
    if agent_name == "llmv2":
        sampling_parameters = SamplingParameters(debug=debug, max_tokens=max_tokens)
        if api_key_file != "":
            api_key_file = open(api_key_file)
            api_key_list = api_key_file.readlines()
            api_key_list = [api_key.strip() for api_key in api_key_list]
        else:
            api_key_list = api_key
        return LLMv2(source=lm_source, lm_id=lm_id, prompt_template_path=prompt_path, cot=True,
                     sampling_parameters=sampling_parameters, task=env_name, api_key=api_key_list,
                     model_and_tokenizer_path=model_and_tokenizer_path)
    elif agent_name == "llm+change":
        sampling_parameters = SamplingParameters(debug=debug)
        if api_key_file != "":
            api_key_file = open(api_key_file)
            api_key_list = api_key_file.readlines()
            api_key_list = [api_key.strip() for api_key in api_key_list]
            return LLMChangeReasoner(source="openai", lm_id=lm_id, prompt_template_path=prompt_path,
                                     cot=True, sampling_parameters=sampling_parameters, task=env_name,
                                     api_key=api_key_list)
        else:
            return LLMChangeReasoner(source="openai", lm_id=lm_id, prompt_template_path=prompt_path,
                                     cot=True, sampling_parameters=sampling_parameters, task=env_name,
                                     api_key=api_key)
    elif agent_name == 'mcts':
        return MCTS(task=env_name)
    elif agent_name == 'mctsv2':
        return MCTSv2(task=env_name)
    elif agent_name == 'rule':
        return RuleBasedAgent(task=env_name)
    elif agent_name == 'human':
        return HumanAgent(task=env_name, prompt_template_path=prompt_path)
    elif agent_name == "greedy":
        return GreedyAgent(task=env_name)
    elif agent_name == "record":
        return RecordAgent(task=env_name)
    elif agent_name == "oracle":
        return OracleAgent(task=env_name)
    elif agent_name == "rl":
        return RLAgent(task=env_name)
    elif agent_name == "random":
        return RandomAgent(task=env_name)
    elif agent_name == "custom":
        return CustomAgent(task=env_name)
    else:
        print(f"Unknown examplar agent: {agent_name}. Please choose from mcts, rule, mctsv2, greedy, rl, random, llm, "
              f"llmv2")
        assert False


def submit(
        output_dir: str = "",  # save path
        env_name: str = "fire",  # 'fire', 'flood' or 'wind',
        agent="",  # a class if you want to use a customized agent, or pass a string to use examplar agent
        data_dir="",  # the path of a data point if you want to evaluate on a selected data point
        port: int = 1071,  # port for TDW build
        max_test_episode: int = 25,  # limitation of the maximum evaluation episodes
        reverse_observation: bool = False,  # if you find the observation is upside-down, please turn on this
        perceptional: bool = False,  # turn on this for the perceptional version of HAZARD
        effect_on_agents: bool = False,  # turn on this to let hazard affect agents
        run_on_test: bool = False,  # turn off to run on test set

        # if you have a network issue with amazon cloud, please turn on this (details in documentation)
        use_cached_assets: bool = False,

        # Parameters for perceptional version of HAZARD
        use_dino: bool = False,  # turn on to use DINO as perception module, instead of mask R-CNN

        # Parameters for examplar LLM or LLMv2 pipelines only
        max_tokens: int = 512,  # maximum max new tokens (only for examplar LLM or LLMv2 pipelines)
        lm_source: str = "openai",  # 'openai' (use OpenAI API) or 'huggingface' (use OpenAI models)
        # for 'openai' examplar LLM or LLMv2 pipelines only
        api_key: str = "",  # OpenAI api key
        api_key_file: str = "",  # instead of api_key, only if you want to use a file containing multiple keys
        lm_id: str = "gpt-3.5-turbo",  # LLM id, such as gpt-3.5-turbo or gpt-4
        # for 'huggingface' examplar LLM or LLMv2 pipelines only
        model_and_tokenizer_path: str = "meta-llama/Llama-2-7b-chat-hf",  # path for HF model you wish to load

        # parameters for making a demo
        record_with_agents: bool = False,  # making demo with an agent
):
    # args = get_args()
    # debug only open when developing the challenge
    debug = False
    # grid size for semantic map
    grid_size = 0.1

    assert env_name in ["fire", "flood", "wind"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if data_dir == "":
        PATH = os.path.dirname(os.path.abspath(__file__))
        while os.path.basename(PATH) != "HAZARD":
            PATH = os.path.dirname(PATH)
        data_dir = DATA_DIR[env_name]
        data_dir = os.path.join(PATH, "data", data_dir)
        if run_on_test:
            data_dir = f"{data_dir}/test_set"

    # initialize the logger
    logger_agent_name = agent if type(agent) == str else "custom"
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    logger = init_logs(output_dir=output_dir, name=f"{env_name}_{logger_agent_name}_{date_time}")

    if type(agent) == str:
        agent_policy = get_examplar_agent(agent_name=agent, api_key=api_key, api_key_file=api_key_file, debug=debug,
                                          max_tokens=max_tokens, lm_source=lm_source, lm_id=lm_id,
                                          model_and_tokenizer_path=model_and_tokenizer_path, env_name=env_name,
                                          prompt_path="policy/llm_configs/prompt_v2.csv")

        if agent == "rl":
            challenge = Challenge(env_name=env_name, data_dir=data_dir, output_dir=output_dir,
                                  logger=logger, launch_build=not debug, debug=debug, port=port, screen_size=1024,
                                  map_size_h=256, map_size_v=256, grid_size=grid_size, use_gt=not perceptional,
                                  reverse_observation=reverse_observation, record_only=False, use_dino=use_dino,
                                  record_with_agents=record_with_agents, effect_on_agents=effect_on_agents,
                                  use_cached_assets=use_cached_assets)
        else:
            challenge = Challenge(env_name=env_name, data_dir=data_dir, output_dir=output_dir,
                                  logger=logger, launch_build=not debug, debug=debug, port=port, screen_size=1024,
                                  grid_size=grid_size, use_gt=not perceptional, record_only=(agent == "record"),
                                  reverse_observation=reverse_observation, record_with_agents=record_with_agents,
                                  use_dino=use_dino, effect_on_agents=effect_on_agents,
                                  use_cached_assets=use_cached_assets)
    else:
        agent_policy = agent
        challenge = Challenge(env_name=env_name, data_dir=data_dir, output_dir=output_dir,
                              logger=logger, launch_build=not debug, debug=debug, port=port, screen_size=1024,
                              grid_size=grid_size, use_gt=not perceptional, record_only=(agent == "record"),
                              reverse_observation=reverse_observation, record_with_agents=record_with_agents,
                              use_dino=use_dino, effect_on_agents=effect_on_agents, use_cached_assets=use_cached_assets)

    if os.path.exists(os.path.join(data_dir, "log.txt")):  # single episode
        challenge.submit(agent=agent_policy, logger=logger, eval_episodes=1)
    else:
        challenge_list = os.listdir(data_dir)
        challenge_list = sorted(challenge_list)
        count = 0
        for task in challenge_list:
            if 'craftroom' not in task and 'kitchen' not in task and 'suburb' not in task:
                continue
            count += 1
            if count > max_test_episode:
                break
            now_data_dir = os.path.join(data_dir, task)
            now_output_dir = os.path.join(output_dir, task)
            if not os.path.exists(now_output_dir):
                os.makedirs(now_output_dir)
            eval_result_path = os.path.join(output_dir, task, 'eval_result.json')
            if os.path.exists(eval_result_path):
                print(f"{eval_result_path} exists")
                continue  # already evaluated
            challenge.output_dir = now_output_dir
            challenge.data_dir = now_data_dir
            print(now_output_dir)
            challenge.submit(agent=agent_policy, logger=logger, eval_episodes=1)
    if challenge.env.controller is not None:
        challenge.env.controller.communicate({"$type": "terminate"})
        challenge.env.controller.socket.close()
