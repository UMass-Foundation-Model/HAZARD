import os.path
import sys
import json

base_dir = "data/room_setup_fire"
locations = {
                1: [13.107263565063477, 1.5, -4.5],
                2: [9.533926010131836, 1.5, -4.0],
                4: [7.040346145629883, 1.5, -0.5],
                5: [10.284771203994751, 1.5, -5.3],
    "mm_craftroom_2a": [2.3700528144836426, 1.5, 0],
    "mm_craftroom_3a": [1.647413730621338, 1.5, 1.5],
    "mm_kitchen_2a": [2.3700528144836426, 1.5, 0],
    "mm_kitchen_3a": [1.647413730621338, 1.5, 1.5],
}

output_dir_list = os.listdir(base_dir)
for output_dir in output_dir_list:
    if output_dir == "test_set":
        output_dir_list1 = os.listdir(os.path.join(base_dir, output_dir))
        for output_dir1 in output_dir_list1:
            out_f = os.path.join(base_dir, output_dir, output_dir1, "flood.json")
            layout = output_dir1.split("-")[0]
            if layout not in locations:
                continue
            location = locations[layout]
            out_fp = open(out_f, "w")
            output_js = {}
            output_js['source'] = [location]
            output_js['direction'] = [[45, 0, 0]]
            output_js['speed'] = [5]
            output_js['flood_source_from'] = 'x_max'
            json.dump(output_js, out_fp)
    else:
        out_f = os.path.join(base_dir, output_dir, "flood.json")
        layout = output_dir.split("-")[0]
        location = locations[layout]
        out_fp = open(out_f, "w")
        output_js = {}
        output_js['source'] = [location]
        output_js['direction'] = [[45, 0, 0]]
        output_js['speed'] = [5]
        output_js['flood_source_from'] = 'x_max'
        json.dump(output_js, out_fp)
