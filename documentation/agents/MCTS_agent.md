## MCTS agent

* Monte Carlo Tree Search (MCTS) is a commonly used algorithm in decision-making problems, which has an effective balance of exploration and exploitation. Since it is hard to get the ground truth frame costs of each action, we design several kinds of heuristic costs for MCTS, such as navigation heuristics, grasp heuristics, drop heuristics, and exploration heuristics. After that, we use MCTS to find an action plan with minimal total cost. 
* Code `policy/mcts.py`
* Usage `--agent_name mcts`