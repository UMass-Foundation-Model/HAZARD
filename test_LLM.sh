env=fire
python run_experiments.py \
--data_dir data/room_setup_fire/test_set \
--api_key_file ~/api_key.txt \
--agent_name llm \
--env_name $env \
--max_tokens 512 \
--prompt_path llm/prompt.csv \
--output_dir outputs/${env}_LLM4/ \
--port 11451 \
--lm_id gpt-4

env=flood
python run_experiments.py \
--data_dir data/room_setup_fire/test_set \
--api_key_file ~/api_key.txt \
--agent_name llm \
--env_name $env \
--max_tokens 512 \
--prompt_path llm/prompt.csv \
--output_dir outputs/${env}_LLM4/ \
--port 11452 \
--lm_id gpt-4

env=wind
python run_experiments.py \
--data_dir data/room_setup_wind/test_set \
--api_key_file ~/api_key.txt \
--agent_name llm \
--env_name $env \
--max_tokens 512 \
--prompt_path llm/prompt.csv \
--output_dir outputs/${env}_LLM4/ \
--port 11453 \
--lm_id gpt-4