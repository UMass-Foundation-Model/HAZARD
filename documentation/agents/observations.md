## Observations

* state
  * state['sem_map']['explored'] (explored map, value is 0 or 1)
  * state['sem_map']['id'] (id map, value is 0 or obj id)
  * state["goal_map"] (value is 0 or -2(agent))
  * state["raw"]["log_temp"] (value is temperature (for fire) or flood height (for flood))
  * ...
* processed_input
  * holding_objects: the object agent holding (list[{'name':name, 'category':category, 'id':id}], length is 0 or 1)
  * nearest_object: the object agent can currently pick up (list[{'name':name, 'category':category, 'id':id}], length is 0 or 1)
  * action_result: result of last action (true/false)
  * step: current step number
  * ...
