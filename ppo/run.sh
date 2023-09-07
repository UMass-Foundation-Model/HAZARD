# export PYTHONPATH=/data/vision/torralba/scratch/chuang/csl/embodied-strategy:$PYTHONPATH
# CUDA_LAUNCH_BLOCKING=1 python main.py --env-name fire-v0 --num-processes 1 --num-env-steps 10 --num-steps 10
export PYTHONPATH=~/csl/embodied-strategy:
# CUDA_LAUNCH_BLOCKING=1 python main_fire.py --env-name fire-v0 --num-processes 1 --log-dir ./fire/ --screen-size 512 --map-size-h 256 --map-size-v 256 --grid-size 0.25
CUDA_LAUNCH_BLOCKING=1 python main_wind.py --env-name wind-v0 --num-processes 1 --log-dir ./wind/ --screen-size 512 --map-size-h 256 --map-size-v 256 --grid-size 0.25
# CUDA_LAUNCH_BLOCKING=1 python main_flood.py --env-name flood-v0 --num-processes 1 --log-dir ./flood/ --screen-size 512 --map-size-h 256 --map-size-v 256 --grid-size 0.25
