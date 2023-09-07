Updated 7.26: 

To remove roof in single room, just add `{"$type": "set_floorplan_roof", "show": false}, ` before `{"$type": "step_physics", "frames": 100}]` at the end of `log.txt`.


Cache usage (take `test_wind.py` for example)

1. ``cd data``
2. ``bash download_assets.sh`` (check its args first, such as platform)
3. Turn on `use_cached_files` in `test_wind.py`

Reset

* target_objects (list[str]) list of target categories
* objects_info (dict[category: {'value':0 low value/1 high value, 'fireproof': 0 low ignition point/ 1 high ignition point, 'waterproof': 0 non-waterproof / 1 waterproof}])

Video recording

* command for running experiment with recording:

```bash
python .\run_experiments.py --data_dir ./data/room_setup_fire/mm_craftroom_2a-1/ --api_key_file ~/api-key.txt --port 1073 --screen_size 1024 --debug --agent_name h_agent --env_name flood
```

* command of generating video:

```bash
ffmpeg -f image2 -framerate 12 -i ./outputs/screenshots/challenge/img_%d.jpg ./outputs/videos/challenge.mp4
```

Observations

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

Actions

* (pick_up, None) or (pick_up, obj id): can only pick up after walk_to
* (drop, None)
* (walk_to, obj id)
* (explore, None): look around
