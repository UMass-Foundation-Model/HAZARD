env=fire
python run_experiments.py \
--data_dir data/room_setup_fire/test_set \
--agent_name h_agent \
--env_name $env \
--output_dir outputs/${env}_h_agent/ \
--port 12345

env=flood
python run_experiments.py \
--data_dir data/room_setup_fire/test_set \
--agent_name h_agent \
--env_name $env \
--output_dir outputs/${env}_h_agent/ \
--port 12346

env=wind
python run_experiments.py \
--data_dir data/room_setup_wind/test_set \
--agent_name h_agent \
--env_name $env \
--output_dir outputs/${env}_h_agent/ \
--port 12347
