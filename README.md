# Official Repo for HAZARD Challenge: Embodied Decision Making in Dynamically Changing Environments

![Overview of HAZARD](pics/overview.png)
An overview of HAZARD task.

## Detailed documentations

### [Get started](documentation/get_started/overview.md)
* [Overview](documentation/get_started/overview.md)
* [Installation](documentation/get_started/install.md)
* [Create your own agent and submit](documentation/get_started/submit.md)
* [Common utils](documentation/get_started/common_utils.md)

### [Agent documents](documentation/agents/agent.md)

* [Observations](documentation/agents/observations.md)

* [Action space](documentation/agents/action_space.md)

* Default agents
  * [RL agent](documentation/agents/rl_agent.md)
  * [MCTS agent](documentation/agents/MCTS_agent.md)
  * [Random agent](documentation/agents/random_agent.md)
  * [Greedy agent](documentation/agents/greedy_agent.md)
  * [Rule-based agent](documentation/agents/rule_based_agent.md)
  * [LLM-based agent](documentation/agents/LLM_agent.md)
    * Version 1.0
    * Version 2.0

* [Customized agent](custom_agent.md)

### Baseline results

* W/o perception (use ground truth segmentations)

  * Fire

| Method | Value rate | Step | Damage |
| -------- | --------  | -------- | -------- |
| Random     |    47.1  |  228.7    |  43.2    |
| RL     |    44.1  |  274.2    |    35.2  |
| Rule     |   45.4   |   243.3   |   32.2   |
| Greedy     |   35.4   |   315.8   |  25.9    |
| MCTS     |   52.5   |   181.1   |   15.7   |
| GPT-3.5     |      |      |      |
| GPT-4     |      |      |      |
| Llama-7b-chat     |      |      |      |
| Llama-13b-chat     |   37.4   |   263.8   |   25.7   |

  * Flood

| Method | Value rate | Step | Damage |
| -------- | --------  | -------- | -------- |
| Random     |   28.7  |    291.3  |   76.8   |
| RL     |    34.5  |  230.9    |   71.7   |
| Rule     |   36.9   |   200.6   |   84.5   |
| Greedy     |  19.5    |  287.9    |  73.3    |
| MCTS     |   32.6   |   170.6   |  65.5    |
| GPT-3.5     |      |      |      |
| GPT-4     |      |      |      |
| Llama-7b-chat     |      |      |      |
| Llama-13b-chat     |   23.8   |   231.6   |   68.5   |

  * Wind

| Method | Value rate | Step |
| -------- | --------  | -------- |
| Random     |   8.4   |  1124.1    |
| RL     |      |      |
| Rule     |   0.0   |  --   |
| Greedy     |      |      |
| MCTS     |   12.9   |   1304.3   |
| GPT-3.5     |      |      |
| GPT-4     |      |      |
| Llama-7b-chat     |      |      |
| Llama-13b-chat     |   10.7   |   1418.5   |

* With perception

  * Fire

| Method | Value rate | Step | Damage |
| -------- | --------  | -------- | -------- |
| Random     |      |      |      |
| RL     |      |      |      |
| Rule ?    |      |      |      |
| Greedy     |      |      |      |
| MCTS     |      |      |      |
| GPT-3.5     |      |      |      |
| GPT-4  ?   |      |      |      |
| Llama-7b-chat     |      |      |      |
| Llama-13b-chat     |      |      |      |

  * Flood

| Method | Value rate | Step | Damage |
| -------- | --------  | -------- | -------- |
| Random     |      |      |      |
| RL     |       |      |      |
| Rule  ?   |       |      |      |
| Greedy ?    |        |      |      |
| MCTS     |        |      |      |
| GPT-3.5     |      |      |      |
| GPT-4  ?   |      |      |      |
| Llama-7b-chat     |      |      |      |
| Llama-13b-chat     |        |      |      |

  * Wind

| Method | Value rate | Step |
| -------- | --------  | -------- |
| Random     |      |      |
| RL     |      |      |
| Rule     |       |      |
| Greedy  ?   |      |      |
| MCTS     |       |      |
| GPT-3.5     |      |      |
| GPT-4     |      |      |
| Llama-7b-chat     |      |      |
| Llama-13b-chat     |      |      |
