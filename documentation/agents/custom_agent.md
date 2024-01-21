## Custom Agent

Please follow the instructions in this page if you want to submit your own agent.
### Agent Implementation
The agent need to be class object with two callable functions: ``reset`` and ``choose_target``

We provide an empty agent framework at `policy/custom.py`. To design your own agent, please reimplement function `self.reset` and `self.choose_target`

### Reset Function
`self.reset()`: this function will be called before each episode. When it is called, agent will receive the following information. The information of all target objects is given in `objects_info`

| Input Parameter    | Type | Description | Example  |  
|-------------|------------------|------------------|------------------|  
| `goal_objects`     | List[str]             | List of target categories |  ["pen", "bowl", "lighter", "key"]       |  
| `objects_info` | Dict             | Keys are category names. Values includes the value, waterproofness or inflammability    | {"pen":{"value": 1, "waterproof": 1}}  |

### Choose Target Function
`self.choose_target`: this function will be called when the trial begins or agent receives the execution result of the last action. This function must return with one action, which will be executed by the environment. It has two dict-type inputs parameters, `state` and `processed_input`, described as the following:

`state`:

| Key    | Value type  | Description                                                                                                                                                                                                                                               |  
|-------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|  
| `raw`     | Dict        | Has 4 keys: rgb, depth, log_temp, and seg_mask. Their values are RGB signals, depth signals, environmental signals (log temperature for fire and water level for flood), and segmentation mask signals (unavailable if you use the perceptional version). |  
| `sem_map` | Dict        | Semantic map built by agent 2D observations + depth. Has 4 keys: height, explored, id, and others. Their values are height map, explored map (0 or 1), object id map, and other information.                                                              |
| `goal_map` | numpy.array | A map of target objects and agent itself. If the agent has spotted the targets, targets will be value 1 in this map. The agent itself will be value -2 in this map. The other elements are 0.                                                             |
| `RL` | Dict        | Information used to train RL agent. Can be ignored here.                                                                                                                                                                                                  |

`procesed_input`:


This function should return a action for agent to execute, in the format of `(action_name, params)`. Agent can either choose a high-level action or a low-level one. The available actions are:

* Available high-level actions

| Action name    | Params | Description | 
|-------------|------------------|------------------|  
| `walk_to`     | Int (target id)             | Agent will try to walk to a target object    |  
| `pick_up` | None or Int (target id)             | Agent will try to pick up a object (if not specified, agent will try to pick up the object agent just walked to)  |
| `drop` | None | If agent has picked up an object, the agent will drop it (in wind task), or put it in its bag (fire or flood task) |
| `explore` | None | Agent will turn by 360 degrees, The observations during turning will update the semantic map of the agent |

* Available low-level actions
Currently only support 4 low-level actions. The params should follow the format of TDW replicant. Please refer the [documentation of TDW Replicant](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/replicants/actions.md) for more details.

| Action name    | Params | Description | 
|-------------|------------------|------------------|  
| `move_by`     | Dict             | Agent will try to move by a certain distance    |  
| `turn_by` | Dict             | Agent will turn by a certain degree  |
| `turn_to` | Dict | Agent will turn to a given degree |
| `reach_for` | Dict | Agent will try to reach for a given position |

To avoid cheating, the following keys in params are forbidden: `max_distance`, `duration`, `scale_duration`, and `arrived_at`.
