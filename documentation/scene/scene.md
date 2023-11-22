## Scene generate

### Generate new scenes
Use `procgen_object_placement.py` in folder `scenes`. 

* command for generating scenes:

  ```
  python ./procgen_object_placement.py --scene mm_craftroom_2a --seed 1234
  ```

Each scene is generated based on a scene from the ProcGenKitchen module. Then some objects are placed on the floor and the top surface of other stuff randomly. The target objects are chosen from these objects. However, setting a random seed in command would not guarantee repeatedly generating the same scene because of TDW implementation. 

### Add new objects
Possible objects are listed in two files: `scenes/scene_configs/list.json` and `scenes/scene_configs/random_object_list_on_floor.txt`. The first file is responsible for objects on other stuffs, while the other one for objects on ground.

We provide an example to add new objects
1. Write the json file of new objects (please follow the format in `scenes/scene_configs/examplar_new_objects.json`)
2. In `scenes/scene_configs/` folder, run `python ./add_new_objects.py examplar_new_objects.json` (please change `examplar_new_objects.json` to your own file)
3. In `scenes/scene_configs/` folder, run `python ./procgen_object_placement.py --scene <choose a room name> --seed 1234 --with_new_objects`
Then scenes with new objects will be generated.

### Assign different attributes

Please edit the configs under `data/meta_data` and `scenes/scene_configs/` for customized attributes.

