### Video recording

* command for running experiment with recording:

```bash
python .\run_experiments.py --data_dir ./data/room_setup_fire/mm_craftroom_2a-1/ --api_key_file ~/api-key.txt --port 1073 --screen_size 1024 --debug --agent_name h_agent --env_name flood
```

* command of generating video:

```bash
ffmpeg -f image2 -framerate 12 -i ./outputs/screenshots/challenge/img_%d.jpg ./outputs/videos/challenge.mp4
```

### Cache usage (take `test_wind.py` for example)
1. ``cd data``
2. ``bash download_assets.sh`` (check its args first, such as platform)
3. Turn on `use_cached_files` in `test_wind.py`

### Display Options
To remove roof in single room, just add `{"$type": "set_floorplan_roof", "show": false}, ` before `{"$type": "step_physics", "frames": 100}]` at the end of `log.txt`.

### Customized settings

* target_objects (list[str]) list of target categories
* objects_info (dict[category: {'value':0 low value/1 high value, 'fireproof': 0 low ignition point/ 1 high ignition point, 'waterproof': 0 non-waterproof / 1 waterproof}])
