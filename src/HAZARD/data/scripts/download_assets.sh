#python download_assets.py \
#       --json_path '../scenes/outputs/commands/2023-04-26 00_16_32.json' \
#       --platform linux \
#       --download_fp_only \
#       --floorplan 2b \
#       --layout 0
#bash download_script.sh
directory="./room_setup_fire/test_set"
for file in "$directory"/*; do
    # Check if the file is a regular file
#    echo $file
    python download_assets.py --json_path "$file/log.txt" --platform linux
    bash download_script.sh
done
