## Installation

### Install ThreeDWorld
HAZARD is built upon ThreeDWorld simulator. Please install ThreeDWorld first.

[ThreeDWorld Installation Guide](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/install.md)

### Install
After installing ThreeDWorld, HAZARD can be installed with pip by
```
git clone https://github.com/zhouqqhh/HAZARD.git
cd HAZARD
pip install -r requirements.txt
pip install .
```
If you want to develop based on this repo, please install it with
```
pip install -e .
```
[Optional] If you have problems building mmcv-full, you can install mmcv with the following command instead of using pip.
```
pip install -U openmim
mim install mmcv>=2.0.0rc1
```

Also, download the rcnn checkpoint from [google drive](https://drive.google.com/file/d/1GFAAkOV5fy_L4c6E7nmtcAxt82BlUbF2/view?usp=sharing) and put it under `src/HAZARD/data`.

### Next step
* [Create your own agent and submit](submit.md)

### Common issues
1. Multiple env registration in gym: update gym.
2. Cannot import HAZARD: add `HAZARD/` to `PYTHONPATH`. In linux, it should be:
```
export PYTHONPATH=$PYTHONPATH:<path to HAZARD>
```
3. Missing imports from mmcv module: follow the official installation guide of mmcv. (You may need to compile it locally if you are using certain versions of CUDA.)
4. To run RL training, install baselines as shown in [rl installation guide](../../ppo/README.md). You may need a separate conda environment if you encounter conflicts.
