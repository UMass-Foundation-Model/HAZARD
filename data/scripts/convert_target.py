import os
import json
import pdb


def generate_target(data_dir):
    info_file = os.path.join(data_dir, "info.json")
    log_file = os.path.join(data_dir, "log.txt")
    info_json = json.load(open(info_file))
    log_lines = open(log_file).readlines()
    log_lines = [json.loads(line) for line in log_lines]
    log_lines = sum(log_lines, [])
    log_objects = [log for log in log_lines if log['$type'] == 'add_object']
    new_target = []
    for target in info_json['targets']:
        for obj in log_objects:
            if obj['name'] == target and obj['category'] not in new_target:
                new_target.append(obj['category'])
    info_json['targets'] = new_target
    json.dump(info_json, open(info_file, 'w'))

if __name__ == "__main__":
    for dir in os.listdir(os.path.join("data", "../room_setup_fire")):
        if dir == "test_set":
            for dir1 in os.listdir(os.path.join("data", "../room_setup_fire", dir)):
                generate_target(os.path.join("data", "../room_setup_fire", dir, dir1))
        else:
            generate_target(os.path.join("data", "../room_setup_fire", dir))
