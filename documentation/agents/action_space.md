## Action space
HAZARD support two different types of action set. Users can choose one of them when implementing agent algorithms.

In this page, we introduce all supported actions in the format of `<action_name> (<parameter>): <action description>`
### High-level action set
* `walk_to (obj_id)`: Agent will try to walk to the 
* `pick_up (obj_id)`: Agent will pick up the object with given `obj_id`. If `obj_id` is `None`, agent will try to pick up the object it just walked to.
* `drop (obj_id)`: Agent will drop the object with given `obj_id`. In fire or flood scenario, agent will simply put the object in its bag, while in wind scenario it will drop the object to a shopping cart or to the ground. If `obj_id` is `None`, agent will try to drop the object it holding.
* `explore`: Agent will simply look round without changing the location of itself.

### Low-level action set
(Adopted from [TDW replicant](https://github.com/threedworld-mit/tdw/blob/b3db46406a3ee8b679f90479162189bc2e1eeb6f/Python/tdw/add_ons/replicant.py))

Low-level actions can be used in format of ('low_level.<action_name>', <params_dict>). Possible <action_name>s are listed below:
* `move_by`: 
  * param distance: The target distance. If less than 0, the Replicant will walk backwards. 
  * param reset_arms: If True, reset the arms to their neutral positions while beginning the walk cycle. 
  * param reset_arms_duration: The speed at which the arms are reset in seconds. 
  * param scale_reset_arms_duration: If True, `reset_arms_duration` will be multiplied by `framerate / 60)`, ensuring smoother motions at faster-than-life simulation speeds. 
* `turn_by`:
  * param angle: The target angle in degrees. Positive value = clockwise turn.
* `turn_to`:
  * param target: The target. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array.
* `reach_for`:
  * param target: The target(s). This can be a list (one target per hand) or a single value (the hand's target). If int: An object ID. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array. 
  * param arm: The Arm value(s) that will reach for each target as a single value or a list. Example: `Arm.left` or `[Arm.left, Arm.right]`. 
  * param absolute: If True, the target position is in world space coordinates. If False, the target position is relative to the Replicant. Ignored if `target` is an int. 
  * param offhand_follows: If True, the offhand will follow the primary hand, meaning that it will maintain the same relative position. Ignored if `arm` is a list or `target` is an int.
  * param from_held: If False, the Replicant will try to move its hand to the `target`. If True, the Replicant will try to move its held object to the `target`. This is ignored if the hand isn't holding an object. 
  * param held_point: The bounds point of the held object from which the offset will be calculated. Can be `"bottom"`, `"top"`, etc. For example, if this is `"bottom"`, the Replicant will move the bottom point of its held object to the `target`. This is ignored if `from_held == False` or ths hand isn't holding an object. 

### Action execution result
* `success`: action executed successfully
* `fail`: fail to execute the action with a reason, typically the following ones
  * `max steps reached`: `walk_to` will fail if the navigation algorithm reaches its max step. Typically it happens when the agent is too far away from the target object or it can not find a path to the target object.
  * `target not in vision or memory`: `walk_to` will fail if the given `obj_id` is not in agent's vision or memory.
  * `cannot grasp, maybe too far`: `pick_up` will fail when the target object is too far from the agent.
  * `not holding an object`: `drop` will fail if the agent is not holding any objects.
