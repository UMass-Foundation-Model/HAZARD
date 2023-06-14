# Installation

### System
We tested our code on an Intel i9-9900k Desktop with RTX2080Super. Our pytorch version is 2.0 with CUDA 11.7. Running ThreeDWorld requires a working Xserver. If you are running on a headless machine, please follow the instructions in **TDW Build**. Be aware that not all headless servers provide Xserver privileges.

### Dependencies

We use `pytorch` for training and evaluation, so it's recommended to install it separately from the [official website](https://pytorch.org/).

First install dependencies that comes in a python package.
```bash
pip3 install -r requirements.txt
```

### TDW Build

Follow the instructions in [tdw](https://github.com/threedworld-mit/tdw) to install TDW Build. Our code and playbacks are tested on Linux platforms. If you intend to run on Windows/MacOS, you can manually modify all `log.txt` files, replacing "linux" with "windows" or "osx" respectively in each url.

In our code, we assume `launch_build=True`, so please make sure the following code runs normally and opens up a "TDW" window:
```python
from tdw.controller import Controller
c = Controller(launch_build=True)
c.communicate({"$type": "terminate"})
c.socket.close()
```

It's important to make sure no two TDW instances use the same port. If you encounter any error, please check if there is any zombie TDW process running in the background. Errors that can't reproduce easily are usually caused by port conflicts.

# Reproducing results

### Agents
Run the following command to reproduce results in the paper.

```bash
python run_experiments.py \
  --output_dir {output directory} \
  --agent_name {agent name: rl,random,etc.} \
  --data_dir {data directory} \
  --env_name {environment name: fire,flood,wind} \
  --port {port number} \
  --api_key_file {file containing openai api key} \
  --lm_id {gpt-3.5-turbo or gpt-4}
```

The data directory should be a subfolder of either `data/room_setup_fire` or `data/room_setup_wind`. If it is a single case, the script above runs it. Otherwise it runs all subcases inside this directory. Flood scenes and fire scenes share the same room setups, so you need to specify which type in `--env_name`.

### RL

To train an RL agent, you need to install [openai-baselines](https://github.com/openai/baselines) from source (the release has some missing imports). Then refer to `ppo/run.sh` to see the three commented commands. Before running this script, replace the PYTHONPATH with the absolute path of your repository.

### Others
Some agents may have a different name than claimed in the paper. For example, `random` is actually the rule-based agent where randomness is only on choosing the target.

### Statistics

When tested on a series of scenes, you can use `calc_value.py` to automatically calculate the statistics. A typical output directory includes several folders, each corresponding to a scene name in `data` directory. Under each folder there is an `eval_result.json` which we will use to calculate the results. For example, if they are of the format `outputs/fire/***/eval_result.json`, run
```bash
python calc_value.py outputs/fire fire # the second argument is the scene name
``` 
to get the results.

# Custom scenes

We also support custom scene creation. Detailed instructions will be added soon.