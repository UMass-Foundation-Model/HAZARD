## Scene generate

To generate a new scene, use ___procgen_object_placement.py___ in folder ___scenes___. 

* command for generating scenes:

  ```
  python ./procgen_object_placement.py --scene mm_craftroom_2a --seed 1234
  ```

Each scene is generated based on a scene from the ProcGenKitchen module. Then some objects are placed on the floor and the top surface of other stuff randomly. The target objects are chosen from these objects. However, setting a random seed in command would not guarantee repeatedly generating the same scene because of TDW implementation. 

Possible objects are listed in two files: _scenes/scene_configs/list.json_ and _scenes/scene_configs/random_object_list_on_floor.txt_. The first file is responsible for objects on other stuffs, while the other one for objects on ground. To change objects list, feel free to modify these two files. Properties of objects are in file _scenes/scene_configs/value.json_ that can also be modified. 

