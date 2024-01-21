## Common utils

### Video recording

* Command for running experiment with recording:

Record without agent: you can just set the `agent` parameter of the `submit` function as `record`. For example,
```bash
from HAZARD import submit

submit(output_dir="outputs/", env_name="fire", agent="record", port=1071, max_test_episode=1,)

```

Record with an agent: you can the on the `record_with_agents` command. For example,
```bash
from HAZARD import submit

submit(output_dir="outputs/", env_name="fire", agent="mcts", record_with_agents=True, port=1071, max_test_episode=1,)

```
And the image can be found in the `outputs/` directory.

After you have generated images, you can convert them to a video with the following command:

```bash
ffmpeg -i img_%04d.jpg -vcodec libx264 -pix_fmt yuv420p video.mp4
```

### Cache usage (take `test_wind.py` for example)
ThreeDWorld requires to download assets from aws. If you have a network issue connecting amazon cloud, you can download asset bundles and use the cache utils.
This option need your patience because it can take multiple rounds to download all assets required.
1. Turn on the ``use_cached_assets`` in ``submit`` function.
2. Run the submit and it will exit with a download notice. Then you need to go to ``src/HAZARD/data/assets`` and run ``bash download_assets.sh``
3. Repeat the step 1 and 2 until the program can run (it means all assets on this episode are cached)

### Display Options
To remove roof in single room, just add `{"$type": "set_floorplan_roof", "show": false}, ` before `{"$type": "step_physics", "frames": 100}]` at the end of `log.txt` in the data point.
