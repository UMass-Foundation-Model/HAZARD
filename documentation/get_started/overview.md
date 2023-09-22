## Overview

### Our purpose
We propose a new simulated embodied benchmark, called HAZARD, specifically designed to assess the decision-making abilities of embodied agents in dynamic situations. HAZARD consists of three unexpected disaster scenarios, including fire, flood, and wind, and specifically supports the utilization of large language models (LLMs) to assist common sense reasoning and decision-making.
![overview.png](..%2F..%2Fpics%2Foverview.png)

### Code structure
* Data: `data/`
  * Fire & flood: `data/room_setup_fire`
  * Wind: `data/room_setup_wind`
* Simulator: `envs/`
  * Fire: `envs/fire`
  * flood: `envs/flood`
  * Wind: `envs/wind`
* Documentations: `documentation/`
* Baseline agents: `policy/`
* Custom scene generation: `scenes/`
* Submit agent and run experiments: `run_experiments.py`

### Next step
* [Installation](install.md)
