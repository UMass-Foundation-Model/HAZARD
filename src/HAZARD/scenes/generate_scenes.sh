#for i in {1..25}
#do
#  python procgen_object_placement.py --with_new_objects --seed "${i}"

python data_convert.py --input "./outputs/commands" --output "../data/room_setup_fire/test_set_new_object"
