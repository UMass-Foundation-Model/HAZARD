import json
import pdb
import sys
import os
import argparse
from tdw.librarian import SceneLibrarian
from tdw.controller import Controller
from tdw.add_ons.floorplan import Floorplan

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json_path', default='../scenes/outputs/commands/2023-04-26\ 00_16_32.json',
        help='scene json path')
    parser.add_argument(
        '--platform', type=str, default="windows", choices=['windows', 'linux', 'osx'])
    parser.add_argument(
        '--download_fp_only', action='store_true', default=False)
    parser.add_argument(
        '--floorplan', type=str, default="2b")
    parser.add_argument(
        '--download_dir', type=str, default="assets")
    parser.add_argument(
        '--layout', type=int, default=0)
    args = parser.parse_args()
    return args

def modify_platform(original_url, current_platform):
    if "/windows/" in original_url:
        return original_url.replace("/windows/", f"/{current_platform}/")
    elif "/osx/" in original_url:
        return original_url.replace("/osx/", f"/{current_platform}/")
    elif "/linux/" in original_url:
        return original_url.replace("/linux/", f"/{current_platform}/")
    else:
        assert False

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.download_dir):
        os.mkdir(args.download_dir)
    output_scripts = open("../download_script.sh", "w")
    platform_convert_dict = {
        "osx": "Darwin",
        "windows": "Windows",
        "linux": "Linux"
    }
    # scene_name = f'floorplan_{args.floorplan}'
    # library = 'scenes.json'

    # if args.download_fp_only:
    #     Controller.SCENE_LIBRARIANS[library] = SceneLibrarian(library)
    #     url = Controller.SCENE_LIBRARIANS[library].get_record(scene_name).urls[platform_convert_dict[args.platform]]
    #     output_scripts.write(f"wget {url} -P {args.download_dir}\n")
    #     exit(0)

    commands = json.load(open(args.json_path))

    # Controller.SCENE_LIBRARIANS[library] = SceneLibrarian(library)
    # Controller.SCENE_LIBRARIANS[library].get_record(scene_name).urls[platform_convert_dict[args.platform]] = \
    #     f'{args.download_dir}/{scene_name}'
    # f = Floorplan()
    # f.init_scene(scene=args.floorplan, layout=args.layout)
    # commands.extend(f.commands)

    urls = []
    for command in commands:
        if 'url' in command and 'amazon' in command['url'] and command['url'] not in urls:
            urls.append(command['url'])
            new_url = modify_platform(command['url'], args.platform)
            output_scripts.write(f"wget -nc {new_url} -P {args.download_dir}\n")
