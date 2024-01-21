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

### Next step
* [Create your own agent and submit](submit.md)
