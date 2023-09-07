## Custom Agent

Please follow the instructions in this page if you want to submit your own agent.
### Agent Implementation

We provide an empty agent framework at `policy/custom.py`. To design your own agent, please reimplement function `self.reset` and `self.choose_target`
* `self.reset()`: this function will be called before each episode. When it is called, agent will receive the following information. The information of all target objects is given in `objects_info`

| Input Parameter    | Type | Description | Example  |  
|-------------|------------------|------------------|------------------|  
| `goal_objects`     | List[str]             | List of target categories |  ["pen", "bowl", "lighter", "key"]       |  
| `objects_info` | Dict             | Keys are category names. Values includes the value, waterproofness or inflammability    | {"pen":{"value": 1, "waterproof": 1}}  |

* `self.choose_target`: this function will be called when the trial begins or agent receives the execution result of the last action. This function must return with one action, which will be executed by the environment. It has two dict-type inputs parameters, `state` and `processed_input`, described as the following:
`state`:

| Key    | Value type | Description | Example  |  
|-------------|------------------|------------------|------------------|  
| `raw`     | Dict    |  |  |  
| `sem_map` |     |   |   |
| `goal_map` |     |   |   |
| `RL` |     |   |   |
