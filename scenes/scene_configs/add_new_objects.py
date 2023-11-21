import json
import sys
new_objects = json.load(open(sys.argv[1]))
# new_objects = json.load(open("examplar_new_objects.json"))

original_obj_list_f = open("random_object_list_on_floor.txt")
original_obj_list = original_obj_list_f.readlines()
original_obj_list = [obj.strip() for obj in original_obj_list]
for obj in new_objects:
    if obj['name'] not in original_obj_list:
        original_obj_list.append(obj['name'])
original_obj_list_f_new = open("random_object_list_on_floor_new.txt", "w")
for obj in original_obj_list:
    original_obj_list_f_new.write(obj + "\n")
original_obj_list_f.close()
original_obj_list_f_new.close()

info = json.load(open("list.json"))
for obj in new_objects:
    if obj['name'] not in info['positive'] and obj not in info['negative']:
        key = 'negative' if obj['value'] == 1 else 'positive'
        info[key].append(obj['name'])
json.dump(info, open("list_new.json", "w"))

value = json.load(open("value_name.json"))
for obj in new_objects:
    if obj['name'] not in value:
        value[obj['name']] = obj['value']
json.dump(value, open("value_name_new.json", "w"))

fire_property = json.load(open("../../data/meta_data/temperature.json"))
name_list = [obj['name'] for obj in fire_property]
for obj in new_objects:
    if obj['name'] not in name_list:
        fire_property.append({
            "name": obj['name'],
            "wcategory": obj['category'],
            "temp": obj['ignition']})
json.dump(fire_property, open("../../data/meta_data/temperature.json", "w"))

flood_property = json.load(open("../../data/meta_data/value.json"))
for obj in new_objects:
    if obj['name'] not in flood_property:
        flood_property[obj['name']] = obj['waterproof']
json.dump(flood_property, open("../../data/meta_data/value.json", "w"))
