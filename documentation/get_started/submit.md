# Create Your Own Agent and Submit

### Try examplar agents
In `starter_code.py`, we have provided an example of using examplar agent.
```python
from HAZARD import submit

submit(output_dir="outputs/", env_name="fire", agent="mcts", port=1071, max_test_episode=1,)

```
In this example, the python program will create a MCTS-based examplar agents and submits it to the fire task, and run it for 1 episode.

The full parameters list of `submit` function are listed below":
```python
output_dir: str = "",  # save path
env_name: str = "fire",  # 'fire', 'flood' or 'wind',
agent="",  # a class if you want to use a customized agent, or pass a string to use examplar agent
data_dir="",  # the path of a data point if you want to evaluate on a selected data point
port: int = 1071,  # port for TDW Build, do not choose a busy port
max_test_episode: int = 25,  # limitation of the maximum evaluation episodes
reverse_observation: bool = False,  # if you find the observation is upside-down, please turn on this
perceptional: bool = False,  # turn on this for the perceptional version of HAZARD
effect_on_agents: bool = False,  # turn on this to let hazard affect agents
run_on_test: bool = False,  # turn off to run on test set

# if you have a network issue with amazon cloud, please turn on this, details can be found in the common_utils document
use_cached_assets: bool = False,

# Parameters for perceptional version of HAZARD
use_dino: bool = False,  # turn on to use DINO as perception module, instead of mask R-CNN

# Parameters for examplar LLM or LLMv2 pipelines only
max_tokens: int = 512,  # maximum max new tokens (only for examplar LLM or LLMv2 pipelines)
lm_source: str = "openai",  # 'openai' (use OpenAI API) or 'huggingface' (use Huggingface models)
# for 'openai' examplar LLM or LLMv2 pipelines only
api_key: str = "",  # OpenAI api key
api_key_file: str = "",  # You can also use this instead of the api_key parameter, only if you want to use a file containing multiple keys.
lm_id: str = "gpt-3.5-turbo",  # LLM id, such as gpt-3.5-turbo or gpt-4
# for 'huggingface' examplar LLM or LLMv2 pipelines only
model_and_tokenizer_path: str = "meta-llama/Llama-2-7b-chat-hf",  # path for HF model you wish to load

# parameters for making a demo
record_with_agents: bool = False,  # making demo with an agent
```

### Full List of Available Examplar Agents
```
llmv2: Large Language Model (LLM)-based agent.
mcts: Monte-Carlo Tree Search (MCTS)-based agent.
mctsv2: Another version of the mcts agent, which models the environmental change explicitly.
rule: A rule-based heuristic agent.
greedy: An agent based on the greedy policy, which always go and rescue the closest object.
rl: A RL-based agent which needs trained RL models to use. (we plan to release the models soon)
random: An agent based on random high-level decisions.
```

For more details, please refer to the [agent document](../agents/agent.md)

### Customized Agent
If you want to submit a customized agent, please replace the `agent` parameter with your agent class object. Suppose you have designed your `TestAgent` class under `test_agent.py`, here is an example
```
from HAZARD import submit
from test_agent import TestAgent

my_agent = TestAgent()
submit(output_dir="outputs/", env_name="fire", agent=my_agent, port=1071, max_test_episode=1,)

```
The detailed requirements of a customized agent can be found in [this](../agents/custom_agent.md)

### Evaluation
The following step will calculate the three metrics used in our paper (Value, Step, and Damage) after the `submit` function completes.
```bash
python src/HAZARD/utils/calc_value.py <path to output_dir> <task name>
```
`task name` is selected from fire, wind, or flood.
The program will output four numbers in format of
```
Average rate ...
Average value rate ...
Average step ...
Damage rate ...
```
, where `Average value rate` is the Value metric, `Average step` is the step metric, and `Damage rate` is the Damage metric.
