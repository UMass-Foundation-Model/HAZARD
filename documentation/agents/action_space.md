## Action space
HAZARD support two different types of action set. Users can choose one of them when implementing agent algorithms.

In this page, we introduce all supported actions in the format of `<action_name> (<parameter>): <action description>`
### High-level action set
* `walk_to (obj_id)`: Agent will try to walk to the 
* `pick_up (obj_id)`: Agent will pick up the object with given `obj_id`. If `obj_id` is `None`, agent will try to pick up the object it just walked to.
* `drop (obj_id)`: Agent will drop the object with given `obj_id`. In fire or flood scenario, agent will simply put the object in its bag, while in wind scenario it will drop the object to a shopping cart or to the ground. If `obj_id` is `None`, agent will try to drop the object it holding.
* `explore`: Agent will simply look round without changing the location of itself.

### Low-level action set
* `walk_by`: 
* `turn_by`:
* ...

### Action execution result
* `success`: action executed successfully
* `fail`: fail to execute the action with a reason, typically the following ones
  * `max steps reached`: `walk_to` will fail if the navigation algorithm reaches its max step. Typically it happens when the agent is too far away from the target object or it can not find a path to the target object.
  * `target not in vision or memory`: `walk_to` will fail if the given `obj_id` is not in agent's vision or memory.
  * `cannot grasp, maybe too far`: `pick_up` will fail when the target object is too far from the agent.
  * `not holding an object`: `drop` will fail if the agent is not holding any objects.
