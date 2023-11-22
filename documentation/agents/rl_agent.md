## RL Agent

* We trained reinforcement learning models using Proximal Policy Optimization (PPO). The actions are the same as described in the random agent. We design the function that rewards picking up and dropping correctly while penalizing actions that fail or have no effect. To make the reward function smoother, we also add a factor of the distance from the agent to the nearest object.
* Code `policy/rl.py` and `ppo/`
* Usage `--agent_name rl`