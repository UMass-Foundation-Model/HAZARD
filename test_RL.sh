env=fire
python run_experiments.py \
--data_dir data/room_setup_fire/test_set \
--agent_name rl \
--env_name $env \
--output_dir outputs/${env}_RL/ \
--port 12222

env=flood
python run_experiments.py \
--data_dir data/room_setup_fire/test_set \
--agent_name rl \
--env_name $env \
--output_dir outputs/${env}_RL/ \
--port 12223

env=wind
python run_experiments.py \
--data_dir data/room_setup_wind/test_set \
--agent_name rl \
--env_name $env \
--output_dir outputs/${env}_RL/ \
--port 12224