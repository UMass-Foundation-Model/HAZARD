## Overview

### Our purpose
HAZARD is a simulated embodied benchmark, specifically designed to assess the decision-making abilities of embodied agents in dynamic situations. HAZARD consists of three unexpected disaster scenarios, including fire, flood, and wind, and specifically supports the utilization of large language models (LLMs) to assist common sense reasoning and decision-making.
### Code structure
* Documentations: `documentation/`
* Data: `src/HAZARD/data/`
  * Fire & flood: `src/HAZARD/data/room_setup_fire`
  * Wind: `src/HAZARD/data/room_setup_wind`
* Simulator: `src/HAZARD/envs/`
  * Fire: `src/HAZARD/envs/fire`
  * flood: `src/HAZARD/envs/flood`
  * Wind: `src/HAZARD/envs/wind`
* Baseline agents: `src/HAZARD/policy/`
* Custom scene generation: `src/HAZARD/scenes/`
* Submit agent and run experiments: `src/HAZARD/run_experiments.py`

### Next step
* [Installation](install.md)
