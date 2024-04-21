<br />
<p align="center">
  <h1 align="center">HAZARD Challenge: Embodied Decision Making in Dynamically Changing Environments (ICLR 2024) </h1>
  <p align="center">
Qinhong Zhou*, Sunli Chen*, Yisong Wang, Haozhe Xu, Weihua Du, Hongxin Zhang,
Yilun Du, Joshua B. Tenenbaum, Chuang Gan
  </p>
  <p align="center">
    <a href='https://arxiv.org/abs/2401.12975'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://vis-www.cs.umass.edu/hazard/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
  <p align="center">
    <img src="pics/overview.png" alt="Logo" width="95%">
  </p>
</p>

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
  * [LLM-based agent](documentation/agents/LLM-based%20agent.md)
    * Version 1.0
    * Version 2.0

* [Customized agent](documentation/agents/custom_agent.md)

### Baseline results

* Full log files can be found in [this link](https://drive.google.com/file/d/1Z1xtble1nCpq1tyuiwYD4wzynRi1MwVK/view?usp=sharing)
* W/o perception (use ground truth segmentations)

  * Fire

| Method | Value rate | Step | Damage |
| -------- | --------  | -------- | -------- |
| Random     |   43.8   |   279.1   |   37.3   |
| RL     |   46.1   |      277.3 |    33.6  |
| Rule     |  53.1    |    236.1  |   32.3   |
| Greedy     |   35.4   |   315.8   |  25.9    |
| MCTS     |   75.9   |   150.1   |   19.7   |
| GPT-3.5     |   70.9   |   170.4   |    20.3  |
| GPT-4     |   77.8   |  159.9    |   15.9   |
| Llama-13b-chat     |  70.2    |   173.8   |    24.0  |

  * Flood

| Method | Value rate | Step | Damage |
| -------- | --------  | -------- | -------- |
| Random     |   28.1   |   286.6   |   80.0   |
| RL     |   35.0   |   252.5   |    71.7  |
| Rule     |   27.3   |   325.3   |    82.2  |
| Greedy     |  18.5    |   289.9   |   80.3   |
| MCTS     |   43.7   |   146.6   |   69.9   |
| GPT-3.5     |   44.3   |   156.6   |   63.7   |
| GPT-4     |   45.7   |   142.9   |   64.9   |
| Llama-13b-chat     |   42.6   |  179.6    |  71.2    |

  * Wind

| Method | Value rate | Step   |
| -------- |------------|--------|
| Random     | 7.1        | 1131.8 |
| RL     | 12.4       | 889.5  |
| Rule     | 0.0        | -      |
| Greedy     | 0.2        | 444.0  |
| MCTS     | 18.0       | 16.7   |
| GPT-3.5     | 23.5       | 735.0  |
| GPT-4     | 31.1       | 590.1  |
| Llama-13b-chat     | 9.6         | 1255.6 |

* With perception

  * Fire

| Method | Value rate | Step | Damage |
| ------- | --------  | -------- | -------- |
| Random   |   41.3   |   314.6   |   31.6   |
| RL     |   45.8   |   241.8   |  35.3    |
| Rule    |   34.5   |   356.3   |    33.7  |
| Greedy   |   35.5   |   257.8   |   25.3   |
| MCTS    |   59.2   |   147.3   |   12.3   |
| GPT-3.5   |   63.5   |    166.6  |   13.5   |
| GPT-4   |   67.7   |   158.5   |   16.1   |
| Llama-13b-chat   |   56.2   |  192.6    |   21.4   |

  * Flood

| Method | Value rate | Step | Damage |
| -------- | --------  | -------- | -------- |
| Random     |   26.7   |    313.5  |   75.8   |
| RL     |   33.1   |    256.6  |   77.0   |
| Rule     |   22.6   |   346.2   |    76.2  |
| Greedy     |    21.5   |  250.7    |   68.8   |
| MCTS     |   30.6    |   145.1   |  63.6    |
| GPT-3.5     |   38.5   |   160.0   |   56.5   |
| GPT-4     |   38.2   |    153.8  |    51.3  |
| Llama-13b-chat     |   34.1   |  193.1    |  69.9    |

  * Wind

| Method | Value rate | Step   |
| -------- |------------|--------|
| Random     | 5.0        | 1113.6 |
| RL     | 8.5        | 1044.9 |
| Rule     | 0.0        | -      |
| Greedy     | 0.2        | 442.0  |
| MCTS     | 18.0       | 939.1  |
| GPT-3.5     | 16.2       | 804.9  |
| GPT-4     | 33.9       | 555.8  |
| Llama-13b-chat     | 16.2       | 1090.1 |
